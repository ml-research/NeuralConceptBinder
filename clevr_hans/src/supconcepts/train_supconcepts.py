import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import glob
import matplotlib.pyplot as plt
from rtpt import RTPT
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from torch.optim import lr_scheduler
from sklearn.model_selection import ShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter


import data_supconcepts as data
import model_supconcepts as model
import utils_supconcepts as utils

torch.set_num_threads(6)
CUDA_LAUNCH_BLOCKING=1
def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed."
    )
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--fp-pretrained-ckpt", help="Path to log file of pretrained slot encoder")

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--ap-log", type=int, default=10, help="Number of epochs before logging AP"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        choices=["clevr-state"],
        help="Use MNIST dataset",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")

    parser.add_argument("--data-dir", type=str, help="Directory to data")
    # Slot attention params
    parser.add_argument('--n-slots', default=10, type=int,
                        help='number of slots for slot attention module')
    parser.add_argument('--n-iters-slot-att', default=3, type=int,
                        help='number of iterations in slot attention module')
    parser.add_argument('--n-attr', default=18, type=int,
                        help='number of attributes per object')

    args = parser.parse_args()

    args.device = 'cuda'
    if args.no_cuda:
        args.device = 'cpu'

    assert args.data_dir.endswith(os.path.sep)
    args.conf_version = args.data_dir.split(os.path.sep)[-2]
    args.name = args.name + f"-{args.conf_version}"
    args.base_pth = 'clevr_hans/src/supconcepts/'

    return args


def get_confusion_from_ckpt(net, test_loader, criterion, args, datasplit, writer=None):

    true, pred, true_wrong, pred_wrong = run_test_final(net, test_loader, criterion, writer, args, datasplit)
    precision, recall, accuracy, f1_score = utils.performance_matrix(true, pred)

    # Generate Confusion Matrix
    if writer is not None:
        utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, 'Confusion_matrix_normalize_{}.pdf'.format(
                                  datasplit))
                              )
        utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(writer.log_dir, 'Confusion_matrix_{}.pdf'.format(datasplit)))
    else:
        utils.plot_confusion_matrix(true, pred, normalize=True, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    'Confusion_matrix_normalize_{}.pdf'.format(datasplit)))
        utils.plot_confusion_matrix(true, pred, normalize=False, classes=args.classes,
                              sFigName=os.path.join(os.path.sep.join(args.fp_ckpt.split(os.path.sep)[:-1]),
                                                    'Confusion_matrix_{}.pdf'.format(datasplit)))
    return accuracy


def get_data(args):
    print('Using image data as input')
    dataset_train = data.CLEVR_HANS_EXPL(
        args.data_dir, "train", lexi=True, max_objects=args.n_slots,
    )

    print('Splitting validation set from train set...')
    sss = ShuffleSplit(n_splits=1, test_size=0.25)
    sss.get_n_splits(dataset_train.scenes, dataset_train.img_class_ids)
    train_index, val_index = next(sss.split(dataset_train.scenes, dataset_train.img_class_ids))

    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)

    dataset_test = data.CLEVR_HANS_EXPL(
        args.data_dir, "val", lexi=True, max_objects=args.n_slots,
    )
    dataset_test_unconf = data.CLEVR_HANS_EXPL(
        args.data_dir, "test", lexi=True, max_objects=args.n_slots,
    )

    args.n_imgclasses = dataset_train.n_classes
    args.class_weights = torch.ones(args.n_imgclasses) / args.n_imgclasses
    args.classes = np.arange(args.n_imgclasses)

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=val_sampler,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True
    )
    test_unconf_loader = DataLoader(
        dataset_test_unconf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=True
    )
    print('Data loaded!')
    print(f'N train: {len(train_index)}')
    print(f'N val: {len(val_index)}')
    print(f'N test: {len(test_loader.dataset)}')
    print(f'N test unconf: {len(test_unconf_loader.dataset)}')

    return train_loader, val_loader, test_loader, test_unconf_loader


def run_test_final(net, loader, criterion, writer, args, datasplit):
    net.eval()

    running_corrects = 0
    running_loss=0
    pred_wrong = []
    true_wrong = []
    preds_all = []
    labels_all = []
    with torch.no_grad():

        for i, sample in enumerate(tqdm(loader)):
            # input is either a set or an image
            imgs, target_set, img_class_ids = map(lambda x: x.to(args.device), sample[:3])

            img_class_ids = img_class_ids.long()

            # forward evaluation through the network
            output_cls, output_attr = net.forward(imgs)

            # class prediction
            _, preds = torch.max(output_cls, 1)

            labels_all.extend(img_class_ids.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())

            running_corrects = running_corrects + torch.sum(preds == img_class_ids)
            loss = criterion(output_cls, img_class_ids)
            running_loss += loss.item()
            preds = preds.cpu().numpy()
            target = img_class_ids.cpu().numpy()
            preds = np.reshape(preds, (len(preds), 1))
            target = np.reshape(target, (len(preds), 1))

            for i in range(len(preds)):
                if (preds[i] != target[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(target[i])

        bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

        if writer is not None:
            writer.add_scalar(f"Loss/{datasplit}_loss", running_loss / len(loader), 0)
            writer.add_scalar(f"Acc/{datasplit}_bal_acc", bal_acc, 0)

        return labels_all, preds_all, true_wrong, pred_wrong

def run(net, loader, optimizer, criterion, split, writer, args, train=False, plot=False, epoch=0):
    if train:
        net.img2state_net.eval()
        net.set_cls.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)
    loader = tqdm(
        loader,
        ncols=0,
        desc="{1} E{0:02d}".format(epoch, "train" if train else "val "),
    )
    running_loss = 0
    preds_all = []
    labels_all = []
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
        # input is either a set or an image
        imgs, target_set, img_class_ids = map(lambda x: x.to(args.device), sample[:3])

        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls, output_attr = net.forward(imgs)

        # class prediction
        _, preds = torch.max(output_cls, 1)

        loss = criterion(output_cls, img_class_ids)

        # Outer optim step
        if train:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        running_loss += loss.item()
        labels_all.extend(img_class_ids.cpu().numpy())
        preds_all.extend(preds.cpu().numpy())

    bal_acc = metrics.balanced_accuracy_score(labels_all, preds_all)

    writer.add_scalar(f"Loss/{split}_loss", running_loss / len(loader), epoch)
    writer.add_scalar(f"Acc/{split}_bal_acc", bal_acc, epoch)

    print("Epoch: {}/{}.. ".format(epoch, args.epochs),
          "{} Loss: {:.3f}.. ".format(split, running_loss / len(loader)),
          "{} Accuracy: {:.3f}.. ".format(split, bal_acc),
          )

    return running_loss / len(loader)


def main():
    args = get_args()

    utils.seed_everything(args.seed)

    train_loader, val_loader, test_loader, test_unconf_loader = get_data(args)

    net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_attr=args.n_attr, n_slots=args.n_slots,
                                   n_iters=args.n_iters_slot_att,
                                   n_set_heads=4, set_transf_hidden=128, category_ids=[3, 6, 8, 10, 17],
                                   device=args.device)
    # update args
    # args.n_attr = net.img2state_net.n_attr

    assert args.fp_pretrained_ckpt
    print(f"Loading ckpt {args.fp_pretrained_ckpt}")
    log = torch.load(args.fp_pretrained_ckpt)
    weights = log["weights"]
    net.img2state_net.load_state_dict(weights, strict=True)

    net = net.to(args.device)

    # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    optimizer = torch.optim.Adam(
        [p for name, p in net.named_parameters() if p.requires_grad and 'set_cls' in name], lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

    torch.backends.cudnn.benchmark = True

    # Create RTPT object
    rtpt = RTPT(name_initials='YOURINITIALS', experiment_name=f"Clevr Hans Slot Att Set Transf xil",
                max_iterations=args.epochs)
    # Start the RTPT tracking
    rtpt.start()

    # tensorboard writer
    writer = utils.create_writer(args)

    cur_best_val_loss = np.inf
    for epoch in range(args.epochs):
        _ = run(net, train_loader, optimizer, criterion, split='train', args=args, writer=writer,
                train=True, plot=False, epoch=epoch)
        # scheduler.step()
        val_loss = run(net, val_loader, optimizer, criterion, split='val', args=args, writer=writer,
                       train=False, plot=True, epoch=epoch)
        _ = run(net, test_loader, optimizer, criterion, split='test', args=args, writer=writer,
                train=False, plot=False, epoch=epoch)

        results = {
            "name": args.name,
            "weights": net.state_dict(),
            "args": args,
        }
        if cur_best_val_loss > val_loss:
            if epoch > 0:
                # remove previous best model
                os.remove(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
            torch.save(results, os.path.join(writer.log_dir, "model_epoch{}_bestvalloss_{:.4f}.pth".format(epoch,
                                                                                                           val_loss)))
            cur_best_val_loss = val_loss

        # Update the RTPT (subtitle is optional)
        rtpt.step()

    print('----------------------------------')
    # load best model for final evaluation
    net = model.NeSyConceptLearner(n_classes=args.n_imgclasses, n_attr=args.n_attr, n_slots=args.n_slots,
                                   n_iters=args.n_iters_slot_att,
                                   n_set_heads=4, set_transf_hidden=128, category_ids=[3, 6, 8, 10, 17],
                                   device=args.device)

    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss_*.pth"))[0])
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    net = net.to(args.device)

    checkpoint = torch.load(glob.glob(os.path.join(writer.log_dir, "model_*_bestvalloss*.pth"))[0])
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    acc = get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best',
                            writer=writer)
    print(f"\nVal. accuracy: {(100*acc):.2f}")
    acc = get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best',
                            writer=writer)
    print(f"\nTest accuracy: {(100*acc):.2f}")

    acc = get_confusion_from_ckpt(net, test_unconf_loader, criterion, args=args, datasplit='test_unconf_best',
                            writer=writer)
    print(f"\nTest unconf accuracy: {(100*acc):.2f}")

    writer.close()


if __name__ == "__main__":
    main()

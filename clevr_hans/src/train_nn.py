import sys
sys.path.insert(0, './clevr_hans/src/')
sys.path.insert(0, '.')
import matplotlib
matplotlib.use("Agg")
import os
import torch
import torch.nn as nn
import numpy as np
import glob
from sklearn import metrics
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

import data as data
import model
import utils_clevr_hans as utils
from rtpt import RTPT
from args_clevr_hans import get_args

torch.autograd.set_detect_anomaly(True)

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
torch.set_num_threads(6)

# -----------------------------------------
# - Define basic and data related methods -
# -----------------------------------------
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

# -----------------------------------------
# - Define Train/Test/Validation methods -
# -----------------------------------------
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
            if args.precompute_bind:
                data, img_class_ids = map(lambda x: x.to(args.device), sample)
            else:
                fnames = sample[3]
                data, _, img_class_ids = map(lambda x: x.to(args.device), sample[:3])

            img_class_ids = img_class_ids.long()

            # forward evaluation through the network
            output_cls = net.forward(data)

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
        net.train()
        torch.set_grad_enabled(True)
    else:
        net.eval()
        torch.set_grad_enabled(False)

    iters_per_epoch = len(loader)
    loader = tqdm(
        loader,
        ncols=0,
        desc="{1} E{0:02d}".format(epoch, split),
    )
    running_loss = 0
    preds_all = []
    labels_all = []
    for i, sample in enumerate(loader, start=epoch * iters_per_epoch):
        if args.precompute_bind:
            data, img_class_ids = map(lambda x: x.to(args.device), sample)
        else:
            fnames = sample[3]
            data, _, img_class_ids = map(lambda x: x.to(args.device), sample[:3])

        img_class_ids = img_class_ids.long()

        # forward evaluation through the network
        output_cls = net.forward(data)

        # class prediction
        _, preds = torch.max(output_cls, 1)

        loss = criterion(output_cls, img_class_ids)

        # Outer optim step
        if train:
            optimizer.zero_grad()
            loss.backward()
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


def train(args):

    train_loader, val_loader, test_loader, test_unconf_loader = utils.get_data(args)

    net = utils.get_pretrained_nn(args)
    net = net.to(args.device)

    # only optimize the set transformer classifier for now, i.e. freeze the state predictor
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
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
        scheduler.step()
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
    net = utils.get_pretrained_nn(args)
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


def test(args):

    print(f"\n\n{args.name} seed {args.seed}\n")
    train_loader, val_loader, test_loader, test_unconf_loader = utils.get_data(args)

    net = utils.get_pretrained_nn(args)
    net = net.to(args.device)

    checkpoint = torch.load(args.fp_ckpt)
    net.load_state_dict(checkpoint['weights'])
    net.eval()
    print("\nModel loaded from checkpoint for final evaluation\n")

    criterion = nn.CrossEntropyLoss()

    acc = get_confusion_from_ckpt(net, val_loader, criterion, args=args, datasplit='val_best', writer=None)
    print(f"\nVal. accuracy: {(100*acc):.2f}")
    acc = get_confusion_from_ckpt(net, test_loader, criterion, args=args, datasplit='test_best', writer=None)
    print(f"\nTest accuracy: {(100*acc):.2f}")
    acc = get_confusion_from_ckpt(net, test_unconf_loader, criterion, args=args, datasplit='test_unconf_best',
                                  writer=None)
    print(f"\nTest unconf accuracy: {(100*acc):.2f}")


def plot(args):
    pass


def main():
    args = get_args()

    args.model_seed = int(args.checkpoint_path.split('seed')[-1].split('/')[0])
    args.save_dir_encs = f'clevr_hans/src/tmp/{args.data_dir.split(os.path.sep)[-2]}/'
    args.base_pth = 'clevr_hans/src/'

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    # elif args.mode == 'plot':
    #     plot(args)


if __name__ == "__main__":
    main()

import numpy as np
import random
import io
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
from PIL import Image
from sklearn import metrics
from matplotlib import rc
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from captum.attr import IntegratedGradients
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models import vit_b_16, resnet18
from collections import OrderedDict

import data as data

axislabel_fontsize = 8
ticklabel_fontsize = 8
titlelabel_fontsize = 8

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pretrained_nn(args):
    if args.nn_base_model == 'vit':
        net = vit_b_16(weights="IMAGENET1K_V1")
        # add more layers as required
        classifier = nn.Sequential(OrderedDict([
            ('head', nn.Linear(in_features=768, out_features=args.n_imgclasses, bias=True))
        ]))

        net.heads = classifier
    elif args.nn_base_model == 'resnet':
        net = resnet18(weights="IMAGENET1K_V1")
        num_ftrs = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, args.n_imgclasses)
        )
    return net


def get_data(args):
    if not args.precompute_bind:
        print('Using image data as input')
        dataset_train = data.CLEVR_HANS_EXPL(
            args.data_dir, "train", img_size=args.image_size, lexi=True, conf_vers=args.conf_version
        )

        print('Splitting validation set from train set...')
        sss = ShuffleSplit(n_splits=1, test_size=0.25)
        sss.get_n_splits(dataset_train.scenes, dataset_train.img_class_ids)
        train_index, val_index = next(sss.split(dataset_train.scenes, dataset_train.img_class_ids))

        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)

        dataset_test = data.CLEVR_HANS_EXPL(
            args.data_dir, "val", img_size=args.image_size, lexi=True, conf_vers=args.conf_version
        )
        dataset_test_unconf = data.CLEVR_HANS_EXPL(
            args.data_dir, "test", img_size=args.image_size, lexi=True, conf_vers=args.conf_version
        )

        args.n_imgclasses = dataset_train.n_classes
        args.class_weights = torch.ones(args.n_imgclasses) / args.n_imgclasses
        args.classes = np.arange(args.n_imgclasses)

        train_loader = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_sampler
        )
        val_loader = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=val_sampler
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        test_unconf_loader = DataLoader(
            dataset_test_unconf,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        print('Data loaded!')
        print(f'N train: {len(train_index)}')
        print(f'N val: {len(val_index)}')
        print(f'N test: {len(test_loader.dataset)}')
        print(f'N test unconf: {len(test_unconf_loader.dataset)}')

    else:
        print(f'Using precomputed ncb ouput as input from {args.save_dir_encs}')

        X_train = np.load(f'{args.save_dir_encs}train_encs_one_hot_{args.model_seed}.npy')
        y_train = np.load(f'{args.save_dir_encs}train_labels_{args.model_seed}.npy')

        print('Splitting validation set from train set...')
        sss = ShuffleSplit(n_splits=1, test_size=0.25)
        sss.get_n_splits(X_train, y_train)
        train_index, val_index = next(sss.split(X_train, y_train))
        X_train, X_val = X_train[train_index], X_train[val_index]
        y_train, y_val = y_train[train_index], y_train[val_index]
        # X_train, , y_train,  = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)

        X_test = np.load(f'{args.save_dir_encs}val_encs_one_hot_{args.model_seed}.npy')
        y_test = np.load(f'{args.save_dir_encs}val_labels_{args.model_seed}.npy')

        X_test_unconf = np.load(f'{args.save_dir_encs}test_encs_one_hot_{args.model_seed}.npy')
        y_test_unconf = np.load(f'{args.save_dir_encs}test_labels_{args.model_seed}.npy')

        dataset_train = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        dataset_val = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        dataset_test = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        dataset_test_unconf = TensorDataset(torch.Tensor(X_test_unconf), torch.Tensor(y_test_unconf))

        args.n_imgclasses = len(np.unique(y_train))
        args.class_weights = torch.ones(args.n_imgclasses) / args.n_imgclasses
        args.classes = np.arange(args.n_imgclasses)

        train_loader = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        test_unconf_loader = DataLoader(
            dataset_test_unconf,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        print('Data loaded!')
        print(f'N train: {len(train_loader.dataset)}')
        print(f'N val: {len(val_loader.dataset)}')
        print(f'N test: {len(test_loader.dataset)}')
        print(f'N test unconf: {len(test_unconf_loader.dataset)}')

    return train_loader, val_loader, test_loader, test_unconf_loader


def set_neg_feedback(net, args):
    """
    Gathers the user feedback in form of a dictionary (see create_neg_feedback_dicts.ipynb) and makes binary masks
    per class and provides these to the model.
    """
    neg_feedback_fp = args.feedback_path
    try:
        with open(neg_feedback_fp, 'rb') as f:
            neg_feedback_dict = pickle.load(f)
    except:
        raise Exception(f"I cannot find a dictionary containing the negative feedback for this model "
                        f"at {neg_feedback_fp}")
    # now transform the feedback into one hot encodings
    cls_feedback_one_hot = torch.zeros((args.n_imgclasses, net.category_ids[-1]), device=args.device)
    for cls_id in range(args.n_imgclasses):
        # if the feedback dictionary for the class is not empty
        if bool(neg_feedback_dict[cls_id]):
            for block_id in neg_feedback_dict[cls_id].keys():
                for concept_id in neg_feedback_dict[cls_id][block_id]:
                    # set the 'negative' feedback to True at the relevant concept id
                    cls_feedback_one_hot[
                    cls_id,
                    net.category_ids[block_id]:net.category_ids[block_id+1]
                    ][concept_id] = 1
    net.cls_feedback_one_hot = cls_feedback_one_hot
    print('Integrated user feedback on irrelevant concepts!')


def compute_loss_negative_feedback(net, data, img_class_ids, args):
    """
    At the irrelevant concepts as indicated by the user feedback mask in net.cls_feedback_one_hot we gather the amount
    of importance the model gives to these concepts.
    """
    expls = generate_intgrad_captum_table(net.set_cls, data, img_class_ids, args)
    assert expls.shape[1] == 1
    expls = expls.squeeze(dim=1)
    # at irrelevant concepts gather the importance that the model gives for those concepts, based on feedback mask
    importance_irr = torch.sum(expls * net.cls_feedback_one_hot[img_class_ids], dim=1)
    return torch.mean(importance_irr)


def resize_tensor(input_tensors, h, w):
    input_tensors = torch.squeeze(input_tensors, 1)

    for i, img in enumerate(input_tensors):
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = transforms.Resize([h, w])(img_PIL)
        img_PIL = transforms.ToTensor()(img_PIL)
        if i == 0:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    final_output = torch.unsqueeze(final_output, 1)
    return final_output


def norm_saliencies(saliencies):
    saliencies_norm = saliencies.clone()

    for i in range(saliencies.shape[0]):
        if len(torch.nonzero(saliencies[i], as_tuple=False)) == 0:
            saliencies_norm[i] = saliencies[i]
        else:
            saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i])) / \
                                 (torch.max(saliencies[i]) - torch.min(saliencies[i]))

    return saliencies_norm


def generate_intgrad_captum_table(net, input, labels, args):
    labels = labels.to(args.device)
    explainer = IntegratedGradients(net)
    saliencies = explainer.attribute(input, target=labels)
    # remove negative attributions
    saliencies[saliencies < 0] = 0.
    return norm_saliencies(saliencies)


def test_hungarian_matching(attrs=torch.tensor([[[0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]],
                                                [[0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0]]]).type(torch.float),
                            pred_attrs=torch.tensor([[[0.01, 0.1, 0.2, 0.1, 0.2, 0.2, 0.01],
                                                      [0.1, 0.6, 0.8, 0., 0.4, 0.001, 0.9]],
                                                     [[0.01, 0.1, 0.2, 0.1, 0.2, 0.2, 0.01],
                                                      [0.1, 0.6, 0.8, 0., 0.4, 0.001, 0.9]]]).type(torch.float)):
    hungarian_matching(attrs, pred_attrs, verbose=1)


def hungarian_matching(attrs, preds_attrs, verbose=0):
    """
    Receives unordered predicted set and orders this to match the nearest GT set.
    :param attrs:
    :param preds_attrs:
    :param verbose:
    :return:
    """
    assert attrs.shape[1] == preds_attrs.shape[1]
    assert attrs.shape == preds_attrs.shape
    from scipy.optimize import linear_sum_assignment
    matched_preds_attrs = preds_attrs.clone()
    idx_map_ids = []
    for sample_id in range(attrs.shape[0]):
        # using euclidean distance
        cost_matrix = torch.cdist(attrs[sample_id], preds_attrs[sample_id]).detach().cpu()

        idx_mapping = linear_sum_assignment(cost_matrix)
        # convert to tuples of [(row_id, col_id)] of the cost matrix
        idx_mapping = [(idx_mapping[0][i], idx_mapping[1][i]) for i in range(len(idx_mapping[0]))]

        idx_map_ids.append([idx_mapping[i][1] for i in range(len(idx_mapping))])

        for i, (row_id, col_id) in enumerate(idx_mapping):
            matched_preds_attrs[sample_id, row_id, :] = preds_attrs[sample_id, col_id, :]
        if verbose:
            print('GT: {}'.format(attrs[sample_id]))
            print('Pred: {}'.format(preds_attrs[sample_id]))
            print('Cost Matrix: {}'.format(cost_matrix))
            print('idx mapping: {}'.format(idx_mapping))
            print('Matched Pred: {}'.format(matched_preds_attrs[sample_id]))
            print('\n')
            # exit()

    idx_map_ids = np.array(idx_map_ids)
    return matched_preds_attrs, idx_map_ids


def clean_expl_str(expl_str):
    return ' & '.join(expl_str.split(' & ')[:-1])


def create_writer(args):
    writer = SummaryWriter(f"{args.base_pth}runs/{args.conf_version}/{args.name}_seed{args.seed}", purge_step=0)

    writer.add_scalar('Hyperparameters/learningrate', args.lr, 0)
    writer.add_scalar('Hyperparameters/num_epochs', args.epochs, 0)
    writer.add_scalar('Hyperparameters/batchsize', args.batch_size, 0)

    # store args as txt file
    with open(os.path.join(writer.log_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")
    return writer


def performance_matrix(true, pred):
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    # print('Confusion Matrix:\n', metrics.confusion_matrix(true, pred))
    print('Precision: {:.3f} Recall: {:.3f}, Accuracy: {:.3f}: ,f1_score: {:.3f}'.format(precision*100,recall*100,
                                                                                         accuracy*100,f1_score*100))
    return precision, recall, accuracy, f1_score


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None,
                          cmap=plt.cm.Blues, sFigName='confusion_matrix.pdf'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(sFigName)
    return ax



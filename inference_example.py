import torch
import sysbinder
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
from rtpt import RTPT
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics

from sysbinder.sysbinder import SysBinderImageAutoEncoder

from data import CLEVREasy_1_WithAnnotations, CLEVR4_1_WithAnnotations
from neural_concept_binder import NeuralConceptBinder

# Baseline, Repository needs to be cloned from https://github.com/yfw/nlotm
import utils_ncb as utils_bnr

torch.set_num_threads(40)
OMP_NUM_THREADS = 40
MKL_NUM_THREADS = 40

SEED = 0


def get_args():
    args = utils_bnr.get_parser(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ).parse_args()

    utils_bnr.set_seed(SEED)

    return args


def gather_encs(model, loader, args):
    model.eval()
    torch.set_grad_enabled(True)

    all_labels_multi = []
    all_labels_single = []
    all_codes = []
    for i, sample in tqdm(enumerate(loader)):
        img_locs = sample[-1]
        sample = sample[:-1]
        imgs, _, _, _ = map(
            lambda x: x.to(args.device), sample
        )

        # encode image with whatever model is being used
        codes, probs = model.encode(imgs)

        # make sure only one object/slot per image
        assert codes.shape[0] == args.batch_size
        # codes = codes.squeeze(dim=1)
        all_codes.append(codes.detach().cpu().numpy())

    all_codes = np.concatenate(all_codes, axis=0)

    return all_codes


def main():
    args = get_args()

    # we train the classifier on the original validation set and test on the original test set
    if "CLEVR-Easy-1" in args.data_path:
        train_dataset = CLEVREasy_1_WithAnnotations(
            root=args.data_path,
            phase="val",
            img_size=args.image_size,
            max_num_objs=args.num_slots,
            num_categories=2,
            perc_imgs=args.perc_imgs,
        )
        test_dataset = CLEVREasy_1_WithAnnotations(
            root=args.data_path,
            phase="test",
            img_size=args.image_size,
            max_num_objs=args.num_slots,
            num_categories=2,
            perc_imgs=1.0,
        )
    elif "CLEVR-4-1" in args.data_path:
        train_dataset = CLEVR4_1_WithAnnotations(
            root=args.data_path,
            phase="val",
            img_size=args.image_size,
            max_num_objs=args.num_slots,
            num_categories=4,
            perc_imgs=args.perc_imgs,
        )
        test_dataset = CLEVR4_1_WithAnnotations(
            root=args.data_path,
            phase="test",
            img_size=args.image_size,
            max_num_objs=args.num_slots,
            num_categories=4,
            perc_imgs=1.0,
        )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    print("-------------------------------------------\n")
    print(f"{len(train_dataset)} train samples, {len(test_dataset)} test samples")
    print(
        f"{args.checkpoint_path} loading for {args.model_type} encoding classification"
    )

    if args.model_type == "ncb":
        model = NeuralConceptBinder(args)
    else:
        raise ValueError(f"Model type {args.model_type} not handled in this script!")

    model.to(args.device)

    # Create and start RTPT object
    rtpt = RTPT(
        name_initials="YOURINITIALS", experiment_name=f"NCB", max_iterations=1
    )
    rtpt.start()

    # gather encodings and corresponding labels
    train_encs = gather_encs(model, train_loader, args)
    # test_encs = gather_encs(model, test_loader, args)

    print("Gathered encodings of provided data ...")
    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()

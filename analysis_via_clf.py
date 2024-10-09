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
from nlotm.nlotm import NlotmImageAutoEncoder
import utils_ncb as utils_ncb

torch.set_num_threads(40)
OMP_NUM_THREADS = 40
MKL_NUM_THREADS = 40

SEED = 0


def get_args():
    args = utils_ncb.get_parser(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ).parse_args()

    utils_ncb.set_seed(SEED)

    if args.model_type == "nlotm":
        args.fp16 = False
        args.vq_type = "vq_ema_dcr"
        args.vq_beta = 1.0
        args.commitment_beta = 50.0
        args.slot_init_type = "random"

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
        imgs, masks, annotations, annotations_multihot = map(
            lambda x: x.to(args.device), sample
        )

        if args.model_type == "nlotm":
            model.downstream_data_type = "z"
            model.downstream_type = "z"
            slots, attns_vis, attns, indices = model.get_z_for_clf(imgs)

            codes = []
            for sample_id in range(imgs.shape[0]):
                mask = masks[sample_id]
                tmp = []
                for j in range(4):
                    tmp.append(torch.sum(attns[sample_id, j] * mask))
                # get id of slot with highest attention
                slot_id = torch.argmax(torch.stack(tmp))
                codes.append(indices[sample_id, slot_id])

            codes = torch.stack(codes)
            codes = codes.unsqueeze(1)
            print(codes.shape)

        else:
            # encode image with whatever model is being used
            encs = model.encode(imgs)

            if "sysbind" in args.model_type:
                codes = encs[0]
                # if we wish to use the sysbinder ptototype attention values as code rather than the weighted prototypes
                if args.attention_codes:
                    codes = torch.argmax(
                        encs[3][1], dim=-1
                    )  # [B, N_ObjSlots, N_Blocks, N_BlockPrototypes]
                    codes = codes.reshape(
                        (codes.shape[0], codes.shape[1], -1)
                    )  # [B, N_ObjSlots, N_Blocks*N_BlockPrototypes]
            elif args.model_type == "ncb":
                codes = encs[0]
                # probs = encs[1]

        assert annotations.shape[1] == 1
        # we consider each attribute for an object as one class
        annotations = annotations.squeeze(dim=1)
        all_labels_single.extend(annotations.detach().cpu().numpy())
        all_labels_multi.extend(annotations_multihot.detach().cpu().numpy())
        # make sure only one object/slot per image
        assert codes.shape[0] == args.batch_size and codes.shape[1] == 1
        codes = codes.squeeze(dim=1)
        codes = codes.detach().cpu().numpy()
        all_codes.append(codes)

    all_labels_multi = np.array(all_labels_multi)
    all_labels_single = np.array(all_labels_single)
    all_codes = np.concatenate(all_codes, axis=0)

    return all_codes, all_labels_single, all_labels_multi


def clf_per_cat(train_encs, train_labels, test_encs, test_labels, model, args):
    """
    Per attribute category fit one linear model to predict the attributes of that category from the
    model encodings.
    """
    train_labels = np.transpose(train_labels)
    test_labels = np.transpose(test_labels)
    accs_per_cat = []
    clfs = []
    max_leaf_nodes = [3, 8]
    for cat_id in range(args.num_categories):
        # initialize linear classifier
        if args.clf_type == "dt":
            clf = DecisionTreeClassifier(random_state=0)
            # clf = DecisionTreeClassifier(random_state=0)
        elif args.clf_type == "nb":
            # TODO: something isn'' working here with NB?
            min_categories = get_min_categories_per_block(model, args)
            clf = CategoricalNB(min_categories=min_categories)
        # fit clf on training encodings and labels
        clf.fit(train_encs, train_labels[cat_id])
        # apply to test encodings
        test_pred = clf.predict(test_encs)
        # compute balanced accuracy
        accs_per_cat.append(
            metrics.balanced_accuracy_score(test_labels[cat_id], test_pred)
        )
        clfs.append(clf)

    return accs_per_cat, clfs


def get_min_categories_per_block(model, args):
    min_categories = []
    for block_id in range(args.num_blocks):
        if args.model_type == "ncb":
            min_categories.append(
                len(
                    np.unique(
                        model.retrieval_corpus[block_id]["ids"].detach().cpu().numpy()
                    )
                )
            )
        else:
            min_categories.append(model.num_prototypes)

    return np.array(min_categories)


def main():
    args = get_args()

    # we train the classifier on the original validation set and test on the original test set
    if "CLEVR-Easy-1" in args.data_path:
        train_dataset = CLEVREasy_1_WithAnnotations(
            root=args.data_path,
            phase="val",
            img_size=args.image_size,
            max_num_objs=args.num_slots,
            num_categories=args.num_categories,
            perc_imgs=args.perc_imgs,
        )
        test_dataset = CLEVREasy_1_WithAnnotations(
            root=args.data_path,
            phase="test",
            img_size=args.image_size,
            max_num_objs=args.num_slots,
            num_categories=args.num_categories,
            perc_imgs=1.0,
        )
    elif "CLEVR-4-1" in args.data_path:
        train_dataset = CLEVR4_1_WithAnnotations(
            root=args.data_path,
            phase="val",
            img_size=args.image_size,
            max_num_objs=args.num_slots,
            num_categories=args.num_categories,
            perc_imgs=args.perc_imgs,
        )
        test_dataset = CLEVR4_1_WithAnnotations(
            root=args.data_path,
            phase="test",
            img_size=args.image_size,
            max_num_objs=args.num_slots,
            num_categories=args.num_categories,
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
    elif "sysbind" in args.model_type:
        if "step" in args.model_type or "hard" in args.model_type:
            assert args.binarize == True
        model = SysBinderImageAutoEncoder(args)
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
            try:
                model.load_state_dict(checkpoint["model"])
                model.image_encoder.sysbinder.prototype_memory.attn.temp = checkpoint[
                    "temp"
                ]
            except:
                model.load_state_dict(checkpoint)
                if args.model_type == "sysbind_step":
                    model.image_encoder.sysbinder.prototype_memory.attn.temp = 0.001
                elif args.model_type == "sysbind_hard":
                    model.image_encoder.sysbinder.prototype_memory.attn.temp = 1e-4
                else:
                    model.image_encoder.sysbinder.prototype_memory.attn.temp = 1.0
            args.log_dir = os.path.join(*args.checkpoint_path.split(os.path.sep)[:-1])
            print(f"loaded ...{args.checkpoint_path}")
        else:
            print("Model path for Sysbinder was not found.")
            return
    elif args.model_type == "nlotm":
        print("Loading NLOTM model")
        model = NlotmImageAutoEncoder(args)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Model type {args.model_type} not recognized")

    model.to(args.device)

    # Create and start RTPT object
    rtpt = RTPT(
        name_initials="YOURINITIALS", experiment_name=f"NCB", max_iterations=1
    )
    rtpt.start()

    # gather encodings and corresponding labels
    train_encs, train_labels_single, train_labels_multi = gather_encs(
        model, train_loader, args
    )
    test_encs, test_labels_single, test_labels_multi = gather_encs(
        model, test_loader, args
    )

    if args.clf_type is not None:
        # classify each attribute category with one linear model
        acc, clf = clf_per_cat(
            train_encs, train_labels_single, test_encs, test_labels_single, model, args
        )
        print(acc)
        print(f"Accuracy of {args.checkpoint_path}: {100 * np.round(np.mean(acc), 4)}")

    print("-------------------------------------------\n")


if __name__ == "__main__":
    main()

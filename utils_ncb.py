import torch
import numpy as np
import random
import os
from sklearn import tree
from matplotlib import pyplot as plt
from datetime import datetime
import argparse


def save_args(args, writer):
    # store args as txt file
    with open(os.path.join(writer.log_dir, "args.txt"), "w") as f:
        for arg in vars(args):
            f.write(f"\n{arg}: {getattr(args, arg)}")


def set_seed(seed=42):
    """
    Set random seeds for all possible random processes.
    :param seed: int
    :return:
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    args = Args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    args.log_dir = os.path.join(*args.checkpoint_path.split(os.path.sep)[:-1])

    # set seed
    torch.manual_seed(args.seed)
    return args


class Args:
    seed = 0
    batch_size = 1
    num_workers = 0
    image_size = 128
    image_channels = 3

    checkpoint_path = "logs/sysbind_orig_seed0/best_model.pt"
    data_path = "/workspace/datasets_wolf/CLEVR-Easy-1-old/"
    log_path = "logs/"

    lr_dvae = 3e-4
    lr_enc = 1e-4
    lr_dec = 3e-4
    lr_warmup_steps = 30000
    lr_half_life = 250000
    clip = 0.05
    epochs = 1
    num_iterations = 3
    num_slots = 4
    num_blocks = 8
    cnn_hidden_size = 512
    slot_size = 2048
    mlp_hidden_size = 192
    num_prototypes = 64
    temp = 1.0
    temp_step = False

    vocab_size = 4096
    num_decoder_layers = 8
    num_decoder_heads = 4
    d_model = 192
    dropout = 0.1

    tau_start = 1.0
    tau_final = 0.1
    tau_steps = 30000

    use_dp = False

    # arguments for clustering
    cluster_type = "kmeans"
    num_clusters = 12
    num_show_per_cluster = 20
    num_categories = 2
    clf_label_type = "individual"
    clf_type = "dt"
    model_type = "bnr"
    n_gen_samples = 500

    name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def print_clf_tree(clf, idx, args):
    # with open(f"{args.log_dir}/tree_structure.txt", "a") as f:
    #     n_nodes = clf.tree_.node_count
    #     children_left = clf.tree_.children_left
    #     children_right = clf.tree_.children_right
    #     feature = clf.tree_.feature
    #     threshold = clf.tree_.threshold
    #     values = clf.tree_.value
    #
    #     node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    #     is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    #     stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    #     while len(stack) > 0:
    #         # `pop` ensures each node is only visited once
    #         node_id, depth = stack.pop()
    #         node_depth[node_id] = depth
    #
    #         # If the left and right child of a node is not the same we have a split
    #         # node
    #         is_split_node = children_left[node_id] != children_right[node_id]
    #         # If a split node, append left and right children and depth to `stack`
    #         # so we can loop through them
    #         if is_split_node:
    #             stack.append((children_left[node_id], depth + 1))
    #             stack.append((children_right[node_id], depth + 1))
    #         else:
    #             is_leaves[node_id] = True
    #
    #     print(
    #         "The binary tree structure has {n} nodes and has "
    #         "the following tree structure:\n".format(n=n_nodes),
    #         file=f
    #     )
    #     for i in range(n_nodes):
    #         if is_leaves[i]:
    #             print(
    #                 "{space}node={node} is a leaf node with value={value}.".format(
    #                     space=node_depth[i] * "\t", node=i, value=values[i]
    #                 ),
    #                 file=f
    #             )
    #         else:
    #             print(
    #                 "{space}node={node} is a split node with value={value}: "
    #                 "go to node {left} if X[:, {feature}] <= {threshold} "
    #                 "else to node {right}.".format(
    #                     space=node_depth[i] * "\t",
    #                     node=i,
    #                     left=children_left[i],
    #                     feature=feature[i],
    #                     threshold=threshold[i],
    #                     right=children_right[i],
    #                     value=values[i],
    #                 ),
    #                 file=f
    #             )
    fsave = os.path.join(args.log_dir, f"tree_structure_{idx}.pdf")
    print(f"Tree structure saved to {fsave}")
    tree.plot_tree(clf)
    plt.savefig(fsave)


def get_parser(device):

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--data_path", default="data/*.png")
    parser.add_argument(
        "--perc_imgs",
        type=float,
        default=1.0,
        help="The percent of images which the clf model should receive as training images "
        "(between 0 and 1). The test set is always the full one (i.e. 1. is 100%)",
    )
    parser.add_argument("--log_path", default="../logs/")
    parser.add_argument(
        "--checkpoint_path", default="logs/sysbind_orig_seed0/best_model.pt"
    )
    parser.add_argument(
        "--model_type",
        choices=["ncb", "sysbind", "sysbind_hard", "sysbind_step"],
        help="Specify whether model type. Either original sysbinder (sysbind) or neural concept binder (ncb).",
        default="ncb",
    )
    parser.add_argument("--use_dp", default=False, action="store_true")
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )

    # arguments for linear probing
    parser.add_argument(
        "--num_categories",
        type=int,
        default=3,
        help="How many categories of attributes for classification?",
    )
    parser.add_argument(
        "--clf_type",
        default=None,
        choices=["dt", "rg"],
        help="Specify the linear classifier model. Either decision tree (dt) or ridge regression model "
        "(rg)",
    )

    # Sysbinder arguments
    parser.add_argument("--lr_dvae", type=float, default=3e-4)
    parser.add_argument("--lr_enc", type=float, default=1e-4)
    parser.add_argument("--lr_dec", type=float, default=3e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=30000)
    parser.add_argument("--lr_half_life", type=int, default=250000)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--cnn_hidden_size", type=int, default=512)
    parser.add_argument("--slot_size", type=int, default=2048)
    parser.add_argument("--mlp_hidden_size", type=int, default=192)
    parser.add_argument("--num_prototypes", type=int, default=64)
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="softmax temperature for prototype binding",
    )
    parser.add_argument("--temp_step", default=False, action="store_true")
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--num_decoder_layers", type=int, default=8)
    parser.add_argument("--num_decoder_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_final", type=float, default=0.1)
    parser.add_argument("--tau_steps", type=int, default=30000)
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--binarize",
        default=False,
        action="store_true",
        help="Should the encodings of the sysbinder be binarized?",
    )
    parser.add_argument(
        "--attention_codes",
        default=False,
        action="store_true",
        help="Should the sysbinder prototype attention values be used as encodings?",
    )

    # R&B arguments
    parser.add_argument(
        "--retrieval_corpus_path",
        default="logs/seed_0/block_concept_dicts.pkl",
    )
    parser.add_argument(
        "--deletion_dict_path",
        default=None,
        help="Path to dictionary containing the deletion feedback, e.g., "
             "logs/seed_0/revision/concepts_deletion_dict.pkl.pkl"
    )
    parser.add_argument(
        "--merge_dict_path",
        default=None,
        help="Path to dictionary containing the merge feedback, e.g., "
             "logs/seed_0/revision/concepts_merge_dict.pkl.pkl"
    )
    parser.add_argument(
        "--retrieval_encs",
        default="proto-exem",
        choices=["proto", "exem", "proto-exem"],
        help="Which type of encodings from the retrieval corpus should we use? Prototypes, exemplars or both jointly?"
    )
    parser.add_argument(
        "--majority_vote",
        default=False,
        action="store_true",
        help="If set then the hard binder takes the majority vote of the topk nearest encodings to "
        "select the final cluster id label",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="If majority_vote is set to True, then hard binder selects the topk nearest encodings at "
        "inference to identify the most likely cluster assignment",
    )
    parser.add_argument(
        "--thresh_attn_obj_slots",
        type=float,
        default=0.98,
        help="threshold value for determining the object slots from set of slots, "
        "based on attention weight values (between 0. and 1.)(see neural_concept_binder for usage)."
        "This should be reestimated for every dataset individually if thresh_count_obj_slots is "
        "not set to 0.",
    )
    parser.add_argument(
        "--thresh_count_obj_slots",
        type=int,
        default=-1,
        help="threshold value (>= -1) for determining the number of object slots from a set of slots, "
        "-1 indicates we wish to use all slots, i.e. no preselection is made"
        "0 indicates we just take that slot with the maximum slot attention value,"
        "1 indicates we take the maximum count of high attn weights (based on thresh_attn_ob_slots), "
        "otherwise those slots that contain a number of values above thresh_attn_obj_slots are chosen"
        "(see neural_concept_binder.py for usage)",
    )

    parser.add_argument("--device", default=device)

    return parser

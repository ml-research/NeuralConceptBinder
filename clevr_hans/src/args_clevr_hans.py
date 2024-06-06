import argparse
import os
from datetime import datetime

import utils_clevr_hans as utils

def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",
    )
    parser.add_argument("--mode", type=str, help="train, test, or plot")
    parser.add_argument("--resume", help="Path to log file to resume from")

    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train with"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--l2_grads", type=float, default=1, help="Right for right reason weight"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size to train with"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
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
    parser.add_argument("--fp-ckpt", type=str, default=None, help="checkpoint filepath")
    parser.add_argument("--fp-pretrained-ckpt", type=str, default=None, help="checkpoint filepath")
    parser.add_argument('--precompute-bind', default=False, action='store_true',
                        help='Use precomputed forward pass of retrieval binder.')

    parser.add_argument("--nn_base_model", type=str, help="vit or resnet")

    # Sysbinder arguments
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--lr_dvae', type=float, default=3e-4)
    parser.add_argument('--lr_enc', type=float, default=1e-4)
    parser.add_argument('--lr_dec', type=float, default=3e-4)
    parser.add_argument('--lr_warmup_steps', type=int, default=30000)
    parser.add_argument('--lr_half_life', type=int, default=250000)
    parser.add_argument('--clip', type=float, default=0.05)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--num_slots', type=int, default=4)
    parser.add_argument('--num_blocks', type=int, default=16)
    parser.add_argument('--cnn_hidden_size', type=int, default=512)
    parser.add_argument('--slot_size', type=int, default=2048)
    parser.add_argument('--mlp_hidden_size', type=int, default=192)
    parser.add_argument('--num_prototypes', type=int, default=64)
    parser.add_argument('--temp', type=float, default=1., help='softmax temperature for prototype binding')
    parser.add_argument('--temp_step', default=False, action='store_true')
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--num_decoder_layers', type=int, default=8)
    parser.add_argument('--num_decoder_heads', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_final', type=float, default=0.1)
    parser.add_argument('--tau_steps', type=int, default=30000)
    parser.add_argument('--binarize', default=False, action='store_true',
                        help='Should the encodings of the sysbinder be binarized?')
    parser.add_argument('--attention_codes', default=False, action='store_true',
                        help='Should the sysbinder prototype attention values be used as encodings?')

    # R&B arguments
    parser.add_argument('--checkpoint_path', default='logs/sysbind_orig_seed0/best_model.pt')
    parser.add_argument('--retrieval_corpus_path', default='logs/sysbind_orig_seed0/block_concept_dicts')
    parser.add_argument('--retrieval_encs', default='proto-exem',
                        choices=['proto', 'exem', 'basis', 'proto-exem', 'proto-exem-basis'])
    parser.add_argument('--majority_vote', default=False, action='store_true',
                        help='If set then the retrieval binder takes the majority vote of the topk nearest encodings to '
                             'select the final cluster id label')
    parser.add_argument('--topk', type=int, default=5,
                        help='if majority_vote is set to True, then retrieval binder selects the topk nearest encodings at '
                             'inference to identify the most likely cluster assignment')
    parser.add_argument('--thresh_attn_obj_slots', type=float, default=0.98,
                        help='threshold value for determining the object slots from set of slots, '
                             'based on attention weight values (between 0. and 1.)(see retrievalbinder for usage).'
                             'This should be reestimated for every dataset individually if thresh_count_obj_slots is '
                             'not set to 0.')
    parser.add_argument('--thresh_count_obj_slots', type=int, default=-1,
                        help='threshold value (>= -1) for determining the number of object slots from a set of slots, '
                             '-1 indicates we wish to use all slots, i.e. no preselection is made'
                             '0 indicates we just take that slot with the maximum slot attention value,'
                             '1 indicates we take the maximum count of high attn weights (based on thresh_attn_ob_slots), '
                             'otherwise those slots that contain a number of values above thresh_attn_obj_slots are chosen'
                             '(see retrievalbinder for usage)')
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
    parser.add_argument("--feedback_path", type=str, default=None, help="Filepath to negative feedback dictionary")

    parser.add_argument('--expl_thresh', type=float, default=0.5,
                        help='concept importance threshold value for determining whether a concept is considered '
                             'important on average over several samples.')
    parser.add_argument('--lambda_expl', type=float, default=100,
                        help='scaling factor for explanation loss.')
    args = parser.parse_args()

    # hard set !!!!!!!!!!!!!!!!!!!!!!!!!
    args.n_heads = 4
    args.set_transf_hidden = 128

    assert args.data_dir.endswith(os.path.sep)
    args.conf_version = args.data_dir.split(os.path.sep)[-2]
    args.name = args.name + f"-{args.conf_version}"

    if args.mode == 'test' or args.mode == 'plot':
        assert args.fp_ckpt

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    utils.seed_everything(args.seed)

    return args

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import CategoricalNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_text
import torch
import json
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.gridspec as gridspec
from datetime import datetime
import sys
import math
import os
from tqdm import tqdm

print(sys.path)
sys.path.append("/Users/toniwuest/Documents/Code/SysBindRetrieve/")
print(sys.path)
from clevr_hans.src.supconcepts.model_supconcepts import (
    NeSyConceptLearner,
    SlotAttention_model,
)
from clevr_puzzle.sudoku.sudoku_solver import solve_sample
from neural_concept_binder import NeuralConceptBinder
from neural_concept_binder import SysBinderImageAutoEncoder
from utils_ncb import get_parser

device = "cpu"

OPTIONS = [1, 3, 5, 10]


def get_sudoku_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="DT",
        help="Choose mode label mapping. Options: CLUSTER, DT",
        choices=["CLUSTER", "DT"],
    )
    parser.add_argument(
        "--plot",
        type=bool,
        default=False,
        help="Choose if cluster examples should be plotted",
    )
    parser.add_argument(
        "--sudoku_dir",
        type=str,
        default="clevr_puzzle/sudoku/CLEVRSudokus",
        help="Choose directory of sudoku samples",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Choose path to model",
    )
    parser.add_argument(
        "--n_options",
        type=int,
        default=1,
        help="Choose number of options for each sudoku sample",
    )
    args = parser.parse_args()
    return args


def load_model(
    model_path,
    retrieval_corpus_path,
    num_blocks=8,
    model_type="ncb",
    delete_dict_path=None,
    merge_dict_paths=None,
):
    argsparser = get_parser(device)
    args = argsparser.parse_args(args=[])
    args.checkpoint_path = model_path
    args.retrieval_corpus_path = retrieval_corpus_path
    args.retrieval_encs = "proto-exem"
    args.thresh_count_obj_slots = 0
    args.deletion_dict_path = delete_dict_path
    args.merge_dict_path = merge_dict_paths

    args.num_blocks = num_blocks

    if model_type == "ncb":
        model = NeuralConceptBinder(args)  # automatically loads the model internally
    elif model_type == "sysbind" or model_type == "sysbind-soft":
        model = SysBinderImageAutoEncoder(args)
    elif model_type == "slot-attention-clevr":
        model = SlotAttention_model(4, 3, 18, device="cpu")
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))["weights"]
        )
    elif model_type == "ground-truth":
        return None

    model.to(device)
    return model


def get_max_clusters_per_block(dict_path):
    with open(dict_path, "rb") as f:
        block_concepts = pickle.load(f)
    max_cluster_per_block = []
    for block in block_concepts:
        exemplars = block["exemplars"]
        max_value = len(exemplars["exemplar_ids"])
        max_cluster_per_block.append(max_value)

    return max_cluster_per_block


def load_img_as_tensor(file_path):
    if type(file_path) == str:
        img = imread(file_path)[:, :, :3]  # discard the alpha values
        img = torch.tensor(img, device=device).unsqueeze(
            0
        )  # add an extra dimension for the number of images
        # reorder dimensions for network
        img = torch.swapaxes(img, 2, 3)
        img = torch.swapaxes(img, 1, 2)
        res = img
    else:
        res = []
        for f in file_path:
            img = imread(f)[:, :, :3]  # discard the alpha values
            img = torch.tensor(img, device=device).unsqueeze(
                0
            )  # add an extra dimension for the number of images
            # reorder dimensions for network
            img = torch.swapaxes(img, 2, 3)
            img = torch.swapaxes(img, 1, 2)
            res.append(img)

        res = torch.concatenate(res)

    return res


def get_distance(code1, code2):
    """Calculate the euclidean distance between two codes."""
    sum = 0
    for i in range(8):
        sum += (code2[i] - code1[i]) ** 2

    dis = np.sqrt(sum)
    return dis


def get_codes_for_image(model, path):
    img = load_img_as_tensor(path)
    model.eval()
    code, _ = model.encode(img)
    code = code[0]
    return code


def get_codes_for_sudoku(model, image_paths, option_paths):

    all_option_codes = []
    for option in option_paths:
        option_codes = []
        for img in option:
            codes = get_codes_for_image(model, img)
            code = get_code_of_object(codes)
            if type(code) == torch.Tensor:
                code = code.numpy()
            option_codes.append(code)
        all_option_codes.append(option_codes)

    # create empty 9x9 array
    sudoku_codes = []

    for row in image_paths:
        for img in row:
            if img:
                codes = get_codes_for_image(model, img)
                code = get_code_of_object(codes)
                if type(code) == torch.Tensor:
                    code = code.numpy()
                sudoku_codes.append(code)

    return all_option_codes, sudoku_codes


def load_sudoku(file_path):
    sudoku_json = json.load(open(file_path))
    image_paths = sudoku_json["images"]
    option_paths = sudoku_json["options"]
    puzzle = sudoku_json["puzzle"]
    solution = sudoku_json["solution"]

    return image_paths, option_paths, solution, puzzle


def calculate_distance(code1, code2):
    sum = 0
    for i in range(8):
        # get absolute value of difference
        sum += abs(code2[i] - code1[i])
        # sum += (code2[i] - code1[i]) ** 2

    dis = sum  # np.sqrt(sum)  # sqrt(sum)
    return dis


def get_image_by_index(index, image_paths):
    counter = 0
    for row in image_paths:
        for path in row:
            if path:
                if counter == index:
                    return path
                counter += 1


def map_labels_to_sudoku(labels, image_paths):
    sudoku = []
    for row in image_paths:
        sudoku_row = []
        for path in row:
            if path:
                sudoku_row.append(labels[0])
                labels = labels[1:]
            else:
                sudoku_row.append(0)
        sudoku.append(sudoku_row)

    return sudoku


def get_codes_via_dt(
    option_codes, object_codes, n_options, seed, true_labels, code_size=8
):

    option_codes = np.array([codes[:n_options] for codes in option_codes])
    # flatten option codes
    option_codes = option_codes.squeeze()
    option_codes = option_codes.reshape(-1, code_size)

    object_codes = np.array(object_codes)
    object_codes = object_codes.squeeze()

    # create an array that contains numbers from 1 to 9 each n_options times
    y = np.array([i for i in range(1, 10) for _ in range(n_options)])

    # train decision tree on option codes
    model = DecisionTreeClassifier(random_state=seed)
    model.fit(option_codes, y)

    # print(export_text(model))

    # predict labels for object codes
    labels = model.predict(object_codes)

    # get prediction error
    errors = np.sum(np.not_equal(labels, true_labels))
    # error ratio
    error_ratio = errors / len(labels)

    return labels, errors, error_ratio


def get_codes_via_naive_bayes(
    option_codes,
    object_codes,
    block_concept_dicts,
    n_options,
    seed,
    true_labels,
    num_blocks=8,
):

    option_codes = np.array([codes[:n_options] for codes in option_codes])
    # flatten option codes
    option_codes = option_codes.squeeze()
    option_codes = option_codes.reshape(-1, num_blocks)

    object_codes = np.array(object_codes)
    object_codes = object_codes.squeeze()

    # replace -1 values with highest value
    max_values_per_column = get_max_clusters_per_block(block_concept_dicts)
    max_value = max(max_values_per_column) + 1

    option_codes[option_codes == -1] = max_value
    object_codes[object_codes == -1] = max_value

    # create an array that contains numbers from 1 to 9 each n_options times
    y = np.array([i for i in range(1, 10) for _ in range(n_options)])

    max_values_per_column = [max_value + 1] * num_blocks

    # min_categories = get_min_categories_per_block(model, args)
    clf = CategoricalNB(min_categories=max_values_per_column)

    # fit clf on training encodings and labels
    clf.fit(option_codes, y)
    # apply to test encodings
    labels = clf.predict(object_codes)

    # get prediction error
    errors = np.sum(labels != true_labels)
    # error ratio
    error_ratio = errors / len(labels)

    return labels, errors, error_ratio


def get_code_from_json(scene_dict, sudoku_type):
    object = scene_dict["objects"][0]
    color = object["color"]
    shape = object["shape"]
    size = object["size"]
    material = object["material"]

    # convert colors and shapes to integers
    color_to_int = {
        "blue": 0,
        "brown": 1,
        "cyan": 2,
        "gray": 3,
        "green": 4,
        "purple": 5,
        "red": 6,
        "yellow": 7,
    }
    shape_to_int = {"cube": 0, "cylinder": 1, "sphere": 2}
    size_to_int = {"large": 0, "small": 1}
    material_to_int = {"metal": 0, "rubber": 1}

    if sudoku_type == "CLEVR-Easy":
        code = torch.tensor([[color_to_int[color], shape_to_int[shape]]])

    elif sudoku_type == "CLEVR-4":
        code = torch.tensor(
            [
                [
                    color_to_int[color],
                    shape_to_int[shape],
                    size_to_int[size],
                    material_to_int[material],
                ]
            ]
        )
    else:
        raise ValueError("Invalid sudoku type")

    return code


def evaluate_sudoku(puzzle, solution, retrieved_solution):

    # compare solutions
    sudoku_successfuly_solved = True
    if retrieved_solution is None:
        sudoku_successfuly_solved = False
        errors_from_clustering = None
        errors_from_solution = None
        # TODO: count errors in clustering
    else:
        errors_from_clustering = 0
        errors_from_solution = 0
        for i in range(9):
            for j in range(9):
                if retrieved_solution[i][j] != solution[i][j]:
                    # print("Error at", i, j)
                    # print(retrieved_solution[i][j], solution[i][j])
                    sudoku_successfuly_solved = False

                    if puzzle[i][j] != 0:
                        errors_from_clustering += 1
                    else:
                        errors_from_solution += 1
        # print("Errors from clustering:", errors_from_clustering)
        # print("Errors from solution:", errors_from_solution)

    return sudoku_successfuly_solved, errors_from_clustering, errors_from_solution


def get_code_for_image(model, path, model_type, sudoku_type):

    if model_type == "ground-truth":
        # get json path of img path
        json_path = path.replace("images", "scenes").replace(".png", ".json")
        with open(json_path, "r") as f:
            json_data = json.load(f)
        codes = get_code_from_json(json_data, sudoku_type)
        return codes

    img = load_img_as_tensor(path)
    model.eval()

    if model_type == "sysbind":
        with torch.no_grad():
            encs = model.encode(img)
            codes = encs[0]
    elif model_type == "sysbind-soft":
        with torch.no_grad():
            encs = model.encode(img)
            codes = encs[0]
            codes = encs[3][1]  # [B, N_ObjSlots, N_Blocks, N_BlockPrototypes]
            codes = codes.reshape(
                (codes.shape[0], codes.shape[1], -1)
            )  # [B, N_ObjSlots, N_Blocks*N_BlockPrototypes]
            codes = codes.squeeze(1)
    elif model_type == "ncb":
        with torch.no_grad():
            codes, _ = model.encode(img)
            codes = codes[0]
    elif model_type == "slot-attention-clevr":
        with torch.no_grad():
            codes = model(img)
            codes = model._transform_attrs(codes)
            # get index of code with highest overall values
            idx = torch.argmax(torch.sum(codes, axis=2))
            codes = codes[:, idx]

            if sudoku_type == "CLEVR-Easy":
                # TODO: remove material and size
                shape = codes[:, 3:6]
                # get index of 1
                shape = torch.argmax(shape, axis=1)
                color = codes[:, 10:]
                color = torch.argmax(color, axis=1)
                codes = torch.cat([shape, color])
                # add dimension
                codes = codes.unsqueeze(0)
            else:
                codes = codes[:, 3:]

    assert codes.shape[0] == 1

    # object_code = get_code_of_object(codes)
    object_code = codes

    return object_code


def solve_clevr_sudokus(args):

    model_dir = f"logs/{args.sudoku_type}"
    model_paths = [
        f"{model_dir}/seed_{seed}/best_model.pt" for seed in args.model_seeds
    ]
    if not args.revised:
        block_concept_paths = [
            f"{model_dir}/seed_{seed}/block_concept_dicts.pkl"
            for seed in args.model_seeds
        ]
        if args.model_type == "sysbind":
            image_to_code_paths = [
                f"{model_dir}/seed_{seed}/sudoku_image_to_code_continous.pkl"
                for seed in args.model_seeds
            ]
        elif args.model_type == "sysbind-soft":
            image_to_code_paths = [
                f"{model_dir}/seed_{seed}/sudoku_image_to_code_soft.pkl"
                for seed in args.model_seeds
            ]
        elif args.model_type == "slot-attention-clevr":
            model_paths = [
                f"logs/CLEVR-4/supervised_concepts/slot-attention-clevr-state-{seed}-0/slot-attention-clevr-state-{seed}-0"
                for seed in args.model_seeds
            ]
            image_to_code_paths = [
                f"logs/CLEVR-4/supervised_concepts/slot-attention-clevr-state-{seed}-0/{args.sudoku_type}_sudoku_image_to_code_2_attr_3.pkl"
                for seed in args.model_seeds
            ]
        elif args.model_type == "ground-truth":
            image_to_code_paths = [
                f"{model_dir}/seed_{seed}/sudoku_image_to_code_supervised.pkl"
                for seed in args.model_seeds
            ]

        else:
            image_to_code_paths = [
                f"{model_dir}/seed_{seed}/sudoku_image_to_code.pkl"
                for seed in args.model_seeds
            ]
    else:
        if args.revise_mode == "dt":
            block_concept_paths = [
                f"{model_dir}/seed_{seed}/block_concept_dicts.pkl"
                for seed in args.model_seeds
            ]
            image_to_code_paths = [
                f"{model_dir}/seed_{seed}/human_revision/human_revision_sudoku_image_to_code_final.pkl"
                for seed in args.model_seeds
            ]
            delete_dict_paths = [
                f"{model_dir}/seed_{seed}/human_revision/blocks_to_delete.pkl"
                for seed in args.model_seeds
            ]
            merge_dict_paths = [
                f"{model_dir}/seed_{seed}/human_revision/blocks_to_merge.pkl"
                for seed in args.model_seeds
            ]
        else:
            block_concept_paths = [
                f"{model_dir}/seed_{seed}/block_concept_dicts.pkl"
                for seed in args.model_seeds
            ]
            image_to_code_paths = [
                f"{model_dir}/seed_{seed}/revision/vlm_revision_sudoku_image_to_code_no_delete.pkl"
                for seed in args.model_seeds
            ]
            delete_dict_paths = [
                f"{model_dir}/seed_{seed}/revision/concepts_delete_dict.pkl"
                for seed in args.model_seeds
            ]
            merge_dict_paths = [
                f"{model_dir}/seed_{seed}/revision/concepts_merge_dict.pkl"
                for seed in args.model_seeds
            ]

    if args.sudoku_type == "CLEVR-Easy":
        args.sudoku_dir = f"clevr_puzzle/sudoku/CLEVR-Easy-Sudokus-K{args.K}"
        num_blocks = 8
        if args.model_type == "sysbind":
            code_size = 2048
        elif args.model_type == "sysbind-soft":
            code_size = 512
        elif args.model_type == "slot-attention-clevr":
            code_size = 2
        elif args.model_type == "ground-truth":
            code_size = 2
        else:
            code_size = 8
    elif args.sudoku_type == "CLEVR-4":
        args.sudoku_dir = f"clevr_puzzle/sudoku/CLEVR-4-Sudokus-K{args.K}"
        num_blocks = 16
        if args.model_type == "sysbind":
            code_size = 2048
        elif args.model_type == "sysbind-soft":
            code_size = 1024
        elif args.model_type == "slot-attention-clevr":
            code_size = 15
        elif args.model_type == "ground-truth":
            code_size = 4
        else:
            code_size = 16

    results = {
        0: {},
        1: {},
        2: {},
    }

    error_results = {
        0: {},
        1: {},
        2: {},
    }

    for i, seed in enumerate(args.model_seeds):

        path_to_codes = image_to_code_paths[i]

        # check if image_to_code dict exists
        try:
            with open(path_to_codes, "rb") as f:
                map_image_to_code = pickle.load(f)
        except FileNotFoundError:
            map_image_to_code = {}
            # create codes for all images
            if args.sudoku_type == "CLEVR-Easy":
                image_folder = "CLEVR-Easy-1/sudoku/images"
            elif args.sudoku_type == "CLEVR-4":
                image_folder = "sudoku/images"

            delete_dict_path = delete_dict_paths[i] if args.revised else None
            merge_dict_path = merge_dict_paths[i] if args.revised else None

            # load model
            model = load_model(
                model_paths[i],
                block_concept_paths[i],
                num_blocks=num_blocks,
                model_type=args.model_type,
                delete_dict_path=delete_dict_path,
                merge_dict_paths=merge_dict_path,
            )
            # get codes for all images
            images = os.listdir(image_folder)
            for image in tqdm(images):
                path = os.path.join(image_folder, image)
                code = get_code_for_image(
                    model, path, args.model_type, args.sudoku_type
                )
                map_image_to_code[path] = code.numpy()

            # save code dict
            with open(path_to_codes, "wb") as f:
                pickle.dump(map_image_to_code, f)

        for n_options in OPTIONS:
            results[seed][n_options] = []
            error_results[seed][n_options] = {
                "errors": [],
                "error_ratios": [],
            }

            for dt_seed in tqdm(range(args.clf_seeds)):

                # Solve sudokus
                solved_sudokus = []
                all_errors = []
                all_error_ratios = []
                for sudoku_id in range(0, 1000):

                    image_paths, option_paths, solution, puzzle = load_sudoku(
                        f"{args.sudoku_dir}/json/sudoku_{sudoku_id}.json"
                    )

                    image_paths_flattened = [
                        path for row in image_paths for path in row if path
                    ]

                    if len(option_paths[0]) == 10:
                        if "json" in option_paths[0][9]:
                            option_paths = [
                                [path.replace("json", "png") for path in option]
                                for option in option_paths
                            ]

                    sudoku_codes = [
                        map_image_to_code[path] for path in image_paths_flattened
                    ]

                    option_codes = [
                        [map_image_to_code[path] for path in option]
                        for option in option_paths
                    ]

                    puzzle_flattened = [e for row in puzzle for e in row if e != 0]

                    # get labels for sudoku
                    if args.mode == "DT":
                        labels, errors, error_ratio = get_codes_via_dt(
                            option_codes,
                            sudoku_codes,
                            n_options,
                            dt_seed,
                            puzzle_flattened,
                            code_size=code_size,
                        )

                    elif args.mode == "NB":
                        labels, errors, error_ratio = get_codes_via_naive_bayes(
                            option_codes,
                            sudoku_codes,
                            block_concept_paths[i],
                            n_options,
                            dt_seed,
                            puzzle_flattened,
                            num_blocks=num_blocks,
                        )

                    else:
                        raise ValueError("Invalid mode")

                    if not errors is None:
                        if errors > 10:
                            sudoku_successfuly_solved = False
                        else:
                            symbolic_sudoku = map_labels_to_sudoku(labels, image_paths)
                            retrieved_solution, _ = solve_sample(symbolic_sudoku)
                            (
                                sudoku_successfuly_solved,
                                errors_from_clustering,
                                errors_from_solution,
                            ) = evaluate_sudoku(puzzle, solution, retrieved_solution)

                    solved_sudokus.append(1 if sudoku_successfuly_solved else 0)
                    all_errors.append(errors)
                    all_error_ratios.append(error_ratio)

                # Report ratio of solved Sudokus
                solved_ratio = (sum(solved_sudokus) / len(solved_sudokus)) * 100
                # print(f"Solved Sudokus:{round(solved_ratio, 4)}%")

                results[seed][n_options] = results[seed][n_options] + [solved_ratio]
                error_results[seed][n_options]["errors"] = error_results[seed][
                    n_options
                ]["errors"] + [np.mean(all_errors)]
                error_results[seed][n_options]["error_ratios"] = error_results[seed][
                    n_options
                ]["error_ratios"] + [np.mean(all_error_ratios)]

    print(results)
    # 3x3 array
    all_accs = np.zeros((3, 4))
    all_errors = np.zeros((3, 4))
    all_error_ratios = np.zeros((3, 4))

    for i in args.model_seeds:
        print("Seed ", i)
        accs = [round(np.mean(results[i][n_options]), 6) for n_options in OPTIONS]
        stds = [round(np.std(results[i][n_options]), 2) for n_options in OPTIONS]
        errors = [
            round(np.mean(error_results[i][n_options]["errors"]), 6)
            for n_options in OPTIONS
        ]
        error_ratios = [
            round(np.mean(error_results[i][n_options]["error_ratios"]), 6)
            for n_options in OPTIONS
        ]
        # print(
        #     f"$ {accs[0]} \pm {stds[0]} $ & $ {accs[1]} \pm {stds[1]} $ & $ {accs[2]} \pm {stds[2]} $ & $ {accs[3]} \pm {stds[3]} $"
        # )
        if len(accs) < 4:
            accs = accs + [0]
            errors = errors + [0]
            error_ratios = error_ratios + [0]
        all_accs[i] = accs
        all_errors[i] = errors
        all_error_ratios[i] = error_ratios

    print(all_accs)
    print("Errors:")
    print(all_errors)
    print("Error ratios:")
    print(all_error_ratios)
    # # get mean and std for each options
    mean_accs = np.mean(all_accs, axis=0)
    std_accs = np.std(all_accs, axis=0)

    print(mean_accs, std_accs)
    # round values
    mean_accs = [round(e, 2) for e in mean_accs]
    std_accs = [round(e, 2) for e in std_accs]

    # print(
    #     f"$ {mean_accs[0]} \pm {std_accs[0]} $ & $ {mean_accs[1]} \pm {std_accs[1]} $ & $ {mean_accs[2]} \pm {std_accs[2]} $ & $ {mean_accs[3]} \pm {std_accs[3]} $"
    # )

    # get mean and std for errors
    mean_errors = np.mean(all_errors, axis=0)
    std_errors = np.std(all_errors, axis=0)
    # print(
    #     f"$ {mean_errors[0]} \pm {std_errors[0]} $ & $ {mean_errors[1]} \pm {std_errors[1]} $ & $ {mean_errors[2]} \pm {std_errors[2]} $ & $ {mean_errors[3]} \pm {std_errors[3]} $"
    # )

    # get mean and std for error ratios
    mean_error_ratios = np.mean(all_error_ratios, axis=0)
    std_error_ratios = np.std(all_error_ratios, axis=0)
    # print(
    #     f"$ {mean_error_ratios[0]} \pm {std_error_ratios[0]} $ & $ {mean_error_ratios[1]} \pm {std_error_ratios[1]} $ & $ {mean_error_ratios[2]} \pm {std_error_ratios[2]} $ & $ {mean_error_ratios[3]} \pm {std_error_ratios[3]} $"
    # )

    revised = ""
    if args.revised:
        revised = "_revised_" + args.revise_mode
    # save accs, errors and error_ratios
    with open(
        f"logs/{args.sudoku_type}/results/accs_{args.K}_{args.mode}_{args.model_type}{revised}.pkl",
        "wb",
    ) as f:
        pickle.dump(all_accs, f)

    with open(
        f"logs/{args.sudoku_type}/results/error_results_{args.K}_{args.mode}_{args.model_type}{revised}.pkl",
        "wb",
    ) as f:
        pickle.dump(all_errors, f)

    with open(
        f"logs/{args.sudoku_type}/results/error_ratios_{args.K}_{args.mode}_{args.model_type}{revised}.pkl",
        "wb",
    ) as f:
        pickle.dump(all_error_ratios, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10, choices=[10, 30, 50])
    parser.add_argument(
        "--mode",
        type=str,
        default="NB",
        choices=["DT", "NB"],
    )
    parser.add_argument("--clf_seeds", type=int, default=10)
    parser.add_argument("--model_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--revised", type=bool, default=False)
    parser.add_argument(
        "--sudoku_type",
        type=str,
        default="CLEVR-Easy",
        choices=["CLEVR-Easy", "CLEVR-4"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="sysbind",
        choices=[
            "sysbind",
            "sysbind-soft",
            "slot-attention-clevr",
            "ncb",
            "ground-truth",
        ],
    )
    parser.add_argument("--revise_mode", type=str, default="dt", choices=["dt", "vlm"])
    args = parser.parse_args()

    for K in [30]:
        for model in [
            "ground-truth",
            "sysbind",
            "sysbind-soft",
            "ncb",
            "slot-attention-clevr",
        ]:
            for sudoku_type in ["CLEVR-Easy", "CLEVR-4"]:
                for revised in [False, True]:
                    if revised:
                        if not model == "ncb":
                            continue
                        for mode in ["dt", "vlm"]:
                            print(
                                f"Experiments for K: {K}, model: {model}, type: {sudoku_type}, revised: {revised}, mode: {mode}"
                            )
                            args.K = K
                            args.model_type = model
                            args.sudoku_type = sudoku_type
                            args.revised = revised
                            args.revise_mode = mode
                            args.clf_seeds = 10
                            args.mode = "DT"
                            args.model_seeds = [0, 1, 2]
                            solve_clevr_sudokus(args)
                    else:
                        print(
                            f"Experiments for K: {K}, model: {model}, type: {sudoku_type}, revised: {revised}"
                        )
                        args.K = K
                        args.model_type = model
                        args.sudoku_type = sudoku_type
                        args.revised = revised
                        args.revise_mode = "dt"
                        args.clf_seeds = 10
                        args.mode = "DT"
                        args.model_seeds = [0, 1, 2]
                        solve_clevr_sudokus(args)

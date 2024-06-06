import random
import os
import json
from sudoku_solver import solve_sample
from sudoku import Sudoku
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


class CLEVRSudoku:
    """Class to represent a CLEVR Sudoku puzzle"""

    def __init__(self, puzzle, solution, map_number_to_attributes):
        self.puzzle = puzzle
        self.solution = solution
        self.map_number_to_attributes = map_number_to_attributes
        self.images = None
        self.options = None

    def to_json(self):
        return {
            "puzzle": self.puzzle,
            "solution": self.solution,
            "map_number_to_attributes": self.map_number_to_attributes,
            "images": self.images,
            "options": self.options,
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    def visualize(self, show=True):
        fig, ax = plt.subplots()
        for i in range(9):
            for j in range(9):
                rect = patches.Rectangle(
                    (i, j), 1, 1, linewidth=1, edgecolor="black", facecolor="none"
                )
                ax.add_patch(rect)

                if self.puzzle[i][j] == 0:
                    continue
                image_path = self.images[i][j]
                img = plt.imread(image_path)
                ax.imshow(img, extent=(i, i + 1, j, j + 1))

        # Adjust line width for thicker lines between every third element
        for i in range(4):
            ax.axvline(i * 3, color="black", linewidth=3)
            ax.axhline(i * 3, color="black", linewidth=3)

        # add lines that frame the plot
        ax.axhline(0, color="black", linewidth=6)
        ax.axhline(9, color="black", linewidth=6)
        ax.axvline(0, color="black", linewidth=6)
        ax.axvline(9, color="black", linewidth=6)

        # don't show axes
        ax.axis("off")

        if show:
            plt.show()

    def save_image(self, path):
        self.visualize(show=False)
        plt.savefig(path)
        plt.close()

    def visualize_options(self, show=True):
        fig, ax = plt.subplots()
        for i in range(9):
            image_path = self.options[i]
            img = plt.imread(image_path)
            ax.imshow(img, extent=(i, i + 1, 0, 1))

        ax.set_xlim(0, 9)
        ax.set_ylim(0, 1)

        # don't show axes
        ax.axis("off")

        # show numbers on top of images
        for i in range(9):
            ax.text(
                i + 0.5,
                1.5,
                str(i + 1),
                ha="center",
                va="center",
                fontsize=10,
            )
        if show:
            plt.show()

    def save_options_image(self, path):
        self.visualize_options(show=False)
        plt.savefig(path)
        plt.close()


class CLEVRSudokuCreator:
    """Class to create CLEVR Sudoku puzzles"""

    def __init__(self, data_folder, colors, shapes, size=None, material=None):
        self.colors = colors
        self.shapes = shapes
        self.size = size
        self.material = material
        self.data_folder = data_folder

        self.all_combinations = self.possible_attribute_combinations()
        self.map_attributes_to_scene = (
            self.setup_images() if self.size is None else self.setup_images(mode="four")
        )
        self.fixed_map_attributes_to_scene = self.map_attributes_to_scene.copy()

    def possible_attribute_combinations(self):
        all_combinations = []

        if self.size is not None and self.material is not None:
            for color in self.colors:
                for shape in self.shapes:
                    for size in self.size:
                        for material in self.material:
                            all_combinations.append((color, shape, size, material))
        else:
            for color in self.colors:
                for shape in self.shapes:
                    all_combinations.append((color, shape))

        return all_combinations

    def setup_images(self, mode="Easy"):
        scene_folder = self.data_folder + "/scenes"

        scenes = [file for file in os.listdir(scene_folder) if file.endswith(".json")]

        map_attributes_to_scene = {}
        for combi in self.all_combinations:
            map_attributes_to_scene[combi] = []

        if mode == "Easy":
            skipped_scenes = 0
            for scene in scenes:
                scene_dict = json.load(open(scene_folder + "/" + scene))
                object = scene_dict["objects"][0]
                shape = object["shape"]
                color = object["color"]
                size = object["size"]
                material = object["material"]
                combi = (color, shape)

                if size == "small" or material == "rubber":
                    skipped_scenes += 1
                    continue

                if combi in map_attributes_to_scene:
                    map_attributes_to_scene[combi] += [scene]

            print(f"Skipped {skipped_scenes} scenes")
        else:
            for scene in scenes:
                scene_dict = json.load(open(scene_folder + "/" + scene))
                object = scene_dict["objects"][0]
                shape = object["shape"]
                color = object["color"]
                size = object["size"]
                material = object["material"]
                combi = (color, shape, size, material)

                if combi in map_attributes_to_scene:
                    map_attributes_to_scene[combi] += [scene]

        for combi in map_attributes_to_scene:
            random.shuffle(map_attributes_to_scene[combi])

        return map_attributes_to_scene

    def create_sudoku(self, N=9, K=40):

        while True:
            sudoku = Sudoku(N, K)
            sudoku.fillValues()

            sudoku_puzzle = sudoku.mat
            sudoku_solution = sudoku.solution

            # pick 9 random attribute combinations from all_combinations
            sudoku_attributes = random.sample(self.all_combinations, 9)

            # map number to attributes
            map_number_to_attributes = {}
            for i in range(1, N + 1):
                map_number_to_attributes[i] = sudoku_attributes[i - 1]

            # check how many solutions the sudoku has
            _, n_solutions = solve_sample(sudoku_puzzle)
            if n_solutions == 1:
                break

        sudoku = CLEVRSudoku(sudoku_puzzle, sudoku_solution, map_number_to_attributes)

        self.add_images_to_sudoku(sudoku)
        self.add_options_to_sudoku(sudoku)
        self.reset_map_attributes_to_scene()

        return sudoku

    def get_scene_for_attributes(self, attribute_combi, remove=False):
        """Get a scene for a given attribute combination"""
        if remove:
            # mix up scenes
            # random.shuffle(self.map_attributes_to_scene[attribute_combi])
            # get scene for attribute_combi
            scene = self.map_attributes_to_scene[attribute_combi][0]

            # remove scene from list
            self.map_attributes_to_scene[attribute_combi] = (
                self.map_attributes_to_scene[attribute_combi][1:]
            )
        else:
            # get scene for attribute_combi
            all_scenes = self.map_attributes_to_scene[attribute_combi]

            # sample scene
            scene = random.choice(all_scenes)

        return scene

    def reset_map_attributes_to_scene(self):
        self.map_attributes_to_scene = self.fixed_map_attributes_to_scene.copy()

        for combi in self.map_attributes_to_scene:
            random.shuffle(self.map_attributes_to_scene[combi])

    def get_image_path_for_scene(self, scene):
        image = scene.replace("json", "png")
        return self.data_folder + "/images/" + image

    def add_images_to_sudoku(self, sudoku):

        puzzle = sudoku.puzzle
        # create empty 9x9 list
        images = [[None] * 9 for _ in range(9)]

        for i in range(len(puzzle)):
            for j in range(len(puzzle[i])):

                if puzzle[i][j] == 0:
                    continue

                attribute_combi = sudoku.map_number_to_attributes[puzzle[i][j]]
                scene = self.get_scene_for_attributes(attribute_combi, remove=True)
                image_path = self.get_image_path_for_scene(scene)

                images[i][j] = image_path

        sudoku.images = images

    def add_options_to_sudoku(self, sudoku):
        options = []
        for i in range(9):
            attribute_combi = sudoku.map_number_to_attributes[i + 1]
            attribute_combi_options = []
            # create 5 options for each attribute combination
            for j in range(5):
                scene = self.get_scene_for_attributes(attribute_combi, remove=True)
                image_path = self.get_image_path_for_scene(scene)
                attribute_combi_options.append(image_path)

            options.append(attribute_combi_options)

        sudoku.options = options


if __name__ == "__main__":

    # creator = CLEVRSudokuCreator(
    #     data_folder="CLEVR-Easy-1/sudoku",
    #     colors=["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    #     shapes=["cube", "sphere", "cylinder"],
    # )

    # K = 30
    # dataset_folder = f"CLEVR-Easy-Sudokus-K{K}-Test"

    # # Create folders
    # os.makedirs(f"clevr_puzzle/sudoku/{dataset_folder}/images", exist_ok=True)
    # os.makedirs(f"clevr_puzzle/sudoku/{dataset_folder}/json", exist_ok=True)

    # for i in tqdm(range(1)):

    #     sudoku = creator.create_sudoku(K=K)
    #     sudoku.save_image(f"clevr_puzzle/sudoku/{dataset_folder}/images/sudoku_{i}.png")
    #     sudoku.save(f"clevr_puzzle/sudoku/{dataset_folder}/json/sudoku_{i}.json")
    #     # sudoku.save_options_image(
    #     #     f"clevr_puzzle/sudoku/CLEVRSudokus/images/options_{i}.png"
    #     # )

    creator = CLEVRSudokuCreator(
        data_folder="sudoku",
        colors=["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
        shapes=["cube", "sphere", "cylinder"],
        size=["small", "large"],
        material=["metal", "rubber"],
    )

    K = 50
    dataset_folder = f"CLEVR-4-Sudokus-K{K}"

    # Create folders
    os.makedirs(f"clevr_puzzle/sudoku/{dataset_folder}/images", exist_ok=True)
    os.makedirs(f"clevr_puzzle/sudoku/{dataset_folder}/json", exist_ok=True)

    for i in tqdm(range(1000)):
        if i % 20 == 0:
            print(i)

        sudoku = creator.create_sudoku(K=K)
        sudoku.save_image(f"clevr_puzzle/sudoku/{dataset_folder}/images/sudoku_{i}.png")
        sudoku.save(f"clevr_puzzle/sudoku/{dataset_folder}/json/sudoku_{i}.json")
        # sudoku.save_options_image(
        #     f"clevr_puzzle/sudoku/CLEVRSudokus/images/options_{i}.png"
        # )

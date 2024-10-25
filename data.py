import glob
import json
import time as time
import h5py
from pathlib import Path

from pycocotools import mask as coco_mask

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import TensorDataset, Dataset, Subset
from torch.utils.data.dataset import ConcatDataset
from collections import Counter
from PIL import Image
from skimage.transform import resize

class CLEVREasyWithAnnotations(Dataset):
    def __init__(self, root, phase, img_size, max_num_objs=3, num_categories=3, perc_imgs=1.):
        assert num_categories in [2, 3]
        self.root = os.path.join(root, phase, 'images', '*.png')
        self.img_size = img_size
        self.max_num_objs = max_num_objs
        self.num_categories = num_categories
        self.perc_imgs = perc_imgs
        if num_categories == 3:
            self.num_attributes_flat = 20
        elif num_categories == 2:
            self.num_attributes_flat = 11
        # self.label_type = label_type

        self.total_imgs = sorted(glob.glob(self.root))
        # remove mask file names
        self.total_imgs = [item for item in self.total_imgs if '_mask.png' not in item]

        if 'CLEVR-Easy-1' in root or 'clevr-easy-1' in root:
            pass
        else:
            if phase == 'train':
                self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
            elif phase == 'val':
                self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
            elif phase == 'test':
                self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
            else:
                # full dataset is used
                pass

        self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * self.perc_imgs)]

        self.transform = T.ToTensor()

        # codes
        self.color_codes = {
            "gray": 0,
            "red": 1,
            "blue": 2,
            "green": 3,
            "brown": 4,
            "purple": 5,
            "cyan": 6,
            "yellow": 7,
        }
        self.shape_codes = {
            "cube": 0,
            "sphere": 1,
            "cylinder": 2,
        }

        # mask colors
        self.object_mask_colors = torch.Tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])  # N, C
        # self.max_num_objs = self.object_mask_colors.shape[0]

        self.eps = 0.001
        self.K = 3

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # paths
        img_loc = self.total_imgs[idx]
        p = Path(img_loc)
        mask_loc = p.parent / (p.stem + "_mask.png")
        json_loc = p.parent.parent / "scenes" / (p.stem + ".json")

        # image
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        image = self.transform(image)  # C, H, W

        # masks
        try:
            mask_image = Image.open(mask_loc).convert("RGB")
            mask_image = mask_image.resize((self.img_size, self.img_size))
            mask_image = self.transform(mask_image)  # C, H, W
            masks = (mask_image[None, :, :, :] < self.object_mask_colors[:, :, None, None] + self.eps) & \
                    (mask_image[None, :, :, :] > self.object_mask_colors[:, :, None, None] - self.eps)
            masks = masks.float().prod(1, keepdim=True)  # N, 1, H, W
        except:
            masks = torch.empty(1)

        # annotations
        annotations = torch.zeros(self.max_num_objs, self.num_categories)  # N, G

        class_labels = []
        with open(json_loc) as f:
            data = json.load(f)
            object_list = data["objects"]
            for i, object in enumerate(object_list):
                # shape
                annotations[i, 0] = self.shape_codes[object["shape"]]

                # color
                annotations[i, 1] = self.color_codes[object["color"]]

                if self.num_categories == 3:
                    # position
                    annotations[i, 2] = np.digitize(object['3d_coords'][0],
                                                    np.linspace(-4 - self.eps, 4 + self.eps, self.K + 1)
                                                    ) - 1
                    annotations[i, 2] = annotations[i, 2] * self.K + np.digitize(
                        object['3d_coords'][1],
                        np.linspace(-3 - self.eps, 4 + self.eps, self.K + 1)
                    ) - 1

        class_labels = torch.tensor(class_labels)

        # convert multi-label to multi-hot annotations
        annotations_multihot = self.anns_to_multihot(annotations)

        attr_labels = torch.empty(1)
        # if self.label_type == 'individual':
        for i in range(annotations_multihot.shape[0]):
            attr_labels = torch.where(annotations_multihot[0])[0]

        return (
            image,  # C, H, W
            masks,  # N, 1, H, W
            annotations,  # N, G
            annotations_multihot,
            class_labels,
            attr_labels
        )

    # takes in a 2d tensor (anns) which is [num concepts, values corresponding attributes for each concept] for an image.
    # This tensor is pulled from 'annonations'
    def anns_to_multihot(self, anns, one_obj=False):
        # annotations = [3, 3, 3] = [batch_size, num concepts, values corresponding attributes for each concept]
        # example : annotations[0,0] = [0,0,0] = a gray cube at someposition
        # one-hot version of this is: [3, 3, (3+8+9)] => one hot = [1 0 0 | 1 0 0 0 0 0 0 0 | 1 0 0 0 0 0 0 0 0 0] = gray cube someposition
        # Note we have 3 + 8 + 9 because there are 3 shapes, 8 colors, and 9 postions
        anns_as_multihot = torch.zeros(self.max_num_objs, self.num_attributes_flat)
        # loop over each object
        for i in range(anns.size(0)):
            one_hot = torch.zeros(self.num_attributes_flat)
            ann = anns[i]
            one_hot[int(ann[0])] = 1
            one_hot[int(ann[1]) + 3] = 1
            if self.num_categories == 3:
                one_hot[int(ann[2]) + 11] = 1
            anns_as_multihot[i] = one_hot

        if one_obj:
            anns_as_multihot = anns_as_multihot[0]

        return anns_as_multihot


class CLEVREasy_1_WithAnnotations(CLEVREasyWithAnnotations):
    def __init__(self, root, phase, img_size, max_num_objs=3, num_categories=3, perc_imgs=1.):
        assert 'CLEVR-Easy-1' in root or 'clevr-easy-1' in root
        super().__init__(root, phase, img_size, max_num_objs, num_categories, perc_imgs)

    def get_img_expl_mask(self, scene):
        mask = 0
        for obj in scene['objects']:
            rle = obj['mask']
            mask += coco_mask.decode(rle)
        return mask

    def __getitem__(self, idx):
        # paths
        img_loc = self.total_imgs[idx]
        p = Path(img_loc)
        mask_loc = p.parent / (p.stem + "_mask.png")
        json_loc = p.parent.parent / "scenes" / (p.stem + ".json")

        with open(json_loc) as f:
            data = json.load(f)

        # image
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        image = self.transform(image)  # C, H, W

        # masks
        try:
            mask_image = Image.open(mask_loc).convert("RGB")
            mask_image = mask_image.resize((self.img_size, self.img_size))
            mask_image = self.transform(mask_image)  # C, H, W
            masks = (mask_image[None, :, :, :] < self.object_mask_colors[:, :, None, None] + self.eps) & \
                    (mask_image[None, :, :, :] > self.object_mask_colors[:, :, None, None] - self.eps)
            masks = masks.float().prod(1, keepdim=True)  # N, 1, H, W
        except:
            try:
                masks = self.get_img_expl_mask(data)
            except:
                masks = torch.empty(1)

        # annotations
        annotations = torch.zeros(1, self.num_categories)  # N, G

        # class_labels = []
        object_list = data["objects"]
        assert len(object_list) == 1
        object = object_list[0]
        # shape
        annotations[0, 0] = self.shape_codes[object["shape"]]

        # color
        annotations[0, 1] = self.color_codes[object["color"]]

        if self.num_categories == 3:
            # position
            annotations[0, 2] = np.digitize(object['3d_coords'][0],
                                            np.linspace(-4 - self.eps, 4 + self.eps, self.K + 1)
                                            ) - 1
            annotations[0, 2] = annotations[0, 2] * self.K + np.digitize(
                object['3d_coords'][1],
                np.linspace(-3 - self.eps, 4 + self.eps, self.K + 1)
            ) - 1

        # convert multi-label to multi-hot annotations
        annotations_multihot = self.anns_to_multihot(annotations, one_obj=True)

        return (
            image,  # C, H, W
            masks,  # N, 1, H, W
            annotations,  # N, G
            annotations_multihot,
            img_loc
        )


class CLEVR4_1_WithAnnotations(Dataset):
    def __init__(self, root, phase, img_size, max_num_objs=3, num_categories=4, perc_imgs=1.):
        assert num_categories == 4
        self.root = os.path.join(root, phase, 'images', '*.png')
        self.img_size = img_size
        self.max_num_objs = max_num_objs
        self.num_categories = num_categories
        self.perc_imgs = perc_imgs
        self.num_attributes_flat = 15

        self.total_imgs = sorted(glob.glob(self.root))
        # remove mask file names
        self.total_imgs = [item for item in self.total_imgs if '_mask.png' not in item]

        if '-1' in root:
            pass
        else:
            if phase == 'train':
                self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
            elif phase == 'val':
                self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
            elif phase == 'test':
                self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
            else:
                # full dataset is used
                pass

        self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * self.perc_imgs)]

        self.transform = T.ToTensor()

        # codes
        self.color_codes = {
            "gray": 0,
            "red": 1,
            "blue": 2,
            "green": 3,
            "brown": 4,
            "purple": 5,
            "cyan": 6,
            "yellow": 7,
        }
        self.shape_codes = {
            "cube": 0,
            "sphere": 1,
            "cylinder": 2,
        }
        self.size_codes = {
            "small": 0,
            "large": 1,
        }
        self.material_codes = {
            "metal": 0,
            "rubber": 1
        }

        self.eps = 0.001
        self.K = 3

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # paths
        img_loc = self.total_imgs[idx]
        p = Path(img_loc)
        # mask_loc = p.parent / (p.stem + "_mask.png")
        json_loc = p.parent.parent / "scenes" / (p.stem + ".json")

        with open(json_loc) as f:
            data = json.load(f)

        # image
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        image = self.transform(image)  # C, H, W

        # dummy for now
        masks = torch.empty(1)

        # annotations
        annotations = torch.zeros(1, self.num_categories)  # N, G

        # class_labels = []
        object_list = data["objects"]
        assert len(object_list) == 1
        object = object_list[0]
        # shape
        annotations[0, 0] = self.shape_codes[object["shape"]]

        # color
        annotations[0, 1] = self.color_codes[object["color"]]

        # size
        annotations[0, 2] = self.size_codes[object["size"]]

        # material
        annotations[0, 3] = self.material_codes[object["material"]]

        # convert multi-label to multi-hot annotations
        annotations_multihot = self.anns_to_multihot(annotations, one_obj=True)

        return (
            image,  # C, H, W
            masks,  # N, 1, H, W
            annotations,  # N, G
            annotations_multihot,
            img_loc
        )

    # takes in a 2d tensor (anns) which is [num concepts, values corresponding attributes for each concept] for an image.
    # This tensor is pulled from 'annonations'
    def anns_to_multihot(self, anns, one_obj=False):
        # annotations = [3, 3, 3] = [batch_size, num concepts, values corresponding attributes for each concept]
        # example : annotations[0,0] = [0,0,0] = a gray cube at someposition
        # one-hot version of this is: [3, 3, (3+8+9)] => one hot = [1 0 0 | 1 0 0 0 0 0 0 0 | 1 0 0 0 0 0 0 0 0 0] = gray cube someposition
        # Note we have 3 + 8 + 9 because there are 3 shapes, 8 colors, and 9 postions
        anns_as_multihot = torch.zeros(self.max_num_objs, self.num_attributes_flat)
        # loop over each object
        for i in range(anns.size(0)):
            one_hot = torch.zeros(self.num_attributes_flat)
            ann = anns[i]
            one_hot[int(ann[0])] = 1
            one_hot[int(ann[1]) + 3] = 1
            if self.num_categories == 3:
                one_hot[int(ann[2]) + 11] = 1
            anns_as_multihot[i] = one_hot

        if one_obj:
            anns_as_multihot = anns_as_multihot[0]

        return anns_as_multihot



class CLEVR4_1_WithAnnotations_LeftRight(Dataset):
    def __init__(self, root, phase, img_size, max_num_objs=3, num_categories=5, perc_imgs=1.):
        assert num_categories == 5
        self.root = os.path.join(root, phase, 'images', '*.png')
        self.img_size = img_size
        self.max_num_objs = max_num_objs
        self.num_categories = num_categories
        self.perc_imgs = perc_imgs
        self.num_attributes_flat = 17

        self.total_imgs = sorted(glob.glob(self.root))
        # remove mask file names
        self.total_imgs = [item for item in self.total_imgs if '_mask.png' not in item]

        if '-1' in root:
            pass
        else:
            if phase == 'train':
                self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
            elif phase == 'val':
                self.total_imgs = self.total_imgs[
                                  int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
            elif phase == 'test':
                self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
            else:
                # full dataset is used
                pass

        self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * self.perc_imgs)]

        self.transform = T.ToTensor()

        # codes
        self.color_codes = {
            "gray": 0,
            "red": 1,
            "blue": 2,
            "green": 3,
            "brown": 4,
            "purple": 5,
            "cyan": 6,
            "yellow": 7,
        }
        self.shape_codes = {
            "cube": 0,
            "sphere": 1,
            "cylinder": 2,
        }
        self.size_codes = {
            "small": 0,
            "large": 1,
        }
        self.material_codes = {
            "metal": 0,
            "rubber": 1
        }

        self.eps = 0.001
        self.K = 3

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # paths
        img_loc = self.total_imgs[idx]
        p = Path(img_loc)
        # mask_loc = p.parent / (p.stem + "_mask.png")
        json_loc = p.parent.parent / "scenes" / (p.stem + ".json")

        with open(json_loc) as f:
            data = json.load(f)

        # image
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        image = self.transform(image)  # C, H, W

        # dummy for now
        masks = torch.empty(1)

        # annotations
        annotations = torch.zeros(1, self.num_categories)  # N, G

        # class_labels = []
        object_list = data["objects"]
        assert len(object_list) == 1
        object = object_list[0]
        # shape
        annotations[0, 0] = self.shape_codes[object["shape"]]

        # color
        annotations[0, 1] = self.color_codes[object["color"]]

        # size
        annotations[0, 2] = self.size_codes[object["size"]]

        # material
        annotations[0, 3] = self.material_codes[object["material"]]

        # left, right position
        annotations[0, 4] = self.obj_position_left_right(object['pixel_coords'], object['mask']['size'])

        # convert multi-label to multi-hot annotations
        annotations_multihot = self.anns_to_multihot(annotations, one_obj=True)

        return (
            image,  # C, H, W
            masks,  # N, 1, H, W
            annotations,  # N, G
            annotations_multihot,
            img_loc
        )

    # takes in a 2d tensor (anns) which is [num concepts, values corresponding attributes for each concept] for an image.
    # This tensor is pulled from 'annonations'
    def anns_to_multihot(self, anns, one_obj=False):
        # annotations = [3, 3, 3] = [batch_size, num concepts, values corresponding attributes for each concept]
        # example : annotations[0,0] = [0,0,0] = a gray cube at someposition
        # one-hot version of this is: [3, 3, (3+8+9)] => one hot = [1 0 0 | 1 0 0 0 0 0 0 0 | 1 0 0 0 0 0 0 0 0 0] = gray cube someposition
        # Note we have 3 + 8 + 9 because there are 3 shapes, 8 colors, and 9 postions
        anns_as_multihot = torch.zeros(self.max_num_objs, self.num_attributes_flat)
        # loop over each object
        for i in range(anns.size(0)):
            one_hot = torch.zeros(self.num_attributes_flat)
            ann = anns[i]
            one_hot[int(ann[0])] = 1
            one_hot[int(ann[1]) + 3] = 1
            if self.num_categories == 3:
                one_hot[int(ann[2]) + 11] = 1
            anns_as_multihot[i] = one_hot

        if one_obj:
            anns_as_multihot = anns_as_multihot[0]

        return anns_as_multihot

    def obj_position_left_right(self, obj_pixel_coords, img_size):
        # Left
        if obj_pixel_coords[0] < int(img_size[0] / 2):
            return 0
        # Right
        elif obj_pixel_coords[0] >= int(img_size[0] / 2):
            return 1


class Tetris_1(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, img_size=128, num_categories=5, perc_imgs=1.):
        """
        Args:
            data_dir (str): Path to the directory containing the image, mask, and attribute files.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target (mask) and returns a transformed version.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        self.num_categories = num_categories
        self.perc_imgs = perc_imgs
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png') and '_mask' not in f]

        self.shape_codes = {
            "I": 0, "O": 1, "L": 2, "T": 3, "Z": 4, "S": 5, "J": 6
        }

        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (255, 255, 0), (255, 165, 0), (0, 255, 255), (128, 0, 128)]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))

        # Load mask
        mask_name = img_name.replace('imgs/', 'masks/').replace('image_', 'image_mask_')
        mask_path = os.path.join(self.data_dir, mask_name)
        mask = Image.open(mask_path).convert("L")  # Grayscale mask
        mask = mask.resize((self.img_size, self.img_size))

        # Load attributes
        npz_name = img_name.replace('imgs/', 'attributes/').replace('.png', '.npz')
        npz_path = os.path.join(self.data_dir, npz_name)
        attributes = np.load(npz_path, allow_pickle=True)['tetris_data']

        # Apply transforms (if any) to image
        if self.transform:
            image = self.transform(image)

        # Apply transforms (if any) to mask
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = T.ToTensor()(mask)  # Default conversion to tensor if no transform

        # Process attributes into a tensor form (optional: you can modify this based on your needs)
        attributes_tensor = self.process_attributes(attributes)

        return image, mask, attributes_tensor

    def process_attributes(self, attributes):
        """
        Process the attributes from the .npz file to a format suitable for a tensor.
        This can be modified based on your needs (e.g., encoding specific values).
        """
        assert self.num_categories == 2 # for now we are only processing discrete values
        # annotations
        annotations = torch.zeros(1, self.num_categories)  # N, G

        # class_labels = []
        assert len(attributes) == 1

        # shape
        annotations[0, 0] = self.shape_codes[attributes[0]["shape"]]

        # color
        annotations[0, 1] = self.get_color_id(attributes[0]["color"])

        return annotations

    def get_color_id(self, color):
        color_id = [i for i in range(len(self.colors)) if color == self.colors[i]][0]
        return color_id


def get_isic_2019(datapath, img_size=None, number_nc=None, number_c=None, normalise=True, most_k_informative_img=None,
              informative_indices_filename='output_wr_metric/informative_score_indices_train_set_most_to_least.npy'):
    """
    Load ISIC Skin Cancer 2019 dataset.
    Return train and test set Dataloaders.

    Args:
        batch_size: specifies the batch size.
        train_shuffle: sets the pytorch Dataloader attribute 'shuffle' which
            'have the data reshuffled at every epoch'.
        number_c: limit the number of cancer images.
        number_nc: limit the number of not-cancer images.
        ce_augment: augments the datasets with counterexamples based on the masks (used for CE).
        informative_indices_filename: Filepath to file which stores the indices of the most
            informative instances (use method in explainer.py method to generate file)
    """
    print("\n----------Dataset----------")
    start = time.time()
    try:
        print("  Read in data from .h5 files...")
        with h5py.File(datapath + 'not_cancer_imgs.h5', 'r') as hf:
            if number_nc is not None:
                not_cancer_imgs = hf['not_cancer_imgs'][:number_nc]
            else:
                not_cancer_imgs = hf['not_cancer_imgs'][:]
        with h5py.File(datapath + 'not_cancer_masks.h5', 'r') as hf:
            if number_nc is not None:
                not_cancer_masks = hf['not_cancer_masks'][:number_nc]
            else:
                not_cancer_masks = hf['not_cancer_masks'][:]
        with h5py.File(datapath + 'not_cancer_flags.h5', 'r') as hf:
            # indicating wether an instances have a seg mask (1, else 0)
            if number_nc is not None:
                not_cancer_flags = hf['not_cancer_flags'][:number_nc]
            else:
                not_cancer_flags = hf['not_cancer_flags'][:]
        with h5py.File(datapath + 'cancer_imgs.h5', 'r') as hf:
            if number_c is not None:
                cancer_imgs = hf['cancer_imgs'][:number_c]
            else:
                cancer_imgs = hf['cancer_imgs'][:]
    except:
        raise RuntimeError(
            "No isic .h5 files found. Please run the setup at setup_isic.py file!")

    end = time.time()
    elap = int(end - start)
    print(f"  --> Read in finished: Took {elap} sec!")

    if img_size is not None:
        not_cancer_imgs_resize = np.zeros((not_cancer_imgs.shape[0], not_cancer_imgs.shape[1], img_size, img_size))
        for n, i in enumerate(not_cancer_imgs):
            not_cancer_imgs_resize[n, :, :, :] = resize(not_cancer_imgs[n, :, :, :],
                                                        not_cancer_imgs_resize.shape[1:], anti_aliasing=True)
        not_cancer_masks_resize = np.zeros((not_cancer_masks.shape[0], not_cancer_masks.shape[1], img_size, img_size))
        for n, i in enumerate(not_cancer_masks):
            not_cancer_masks_resize[n, :, :, :] = resize(not_cancer_masks[n, :, :, :],
                                                        not_cancer_masks_resize.shape[1:], anti_aliasing=True)
        cancer_imgs_resize = np.zeros((cancer_imgs.shape[0], cancer_imgs.shape[1], img_size, img_size))
        for n, i in enumerate(cancer_imgs):
            cancer_imgs_resize[n, :, :, :] = resize(cancer_imgs[n, :, :, :],
                                                    cancer_imgs_resize.shape[1:], anti_aliasing=True)
        not_cancer_imgs = not_cancer_imgs_resize
        not_cancer_masks = not_cancer_masks_resize
        cancer_imgs = cancer_imgs_resize

        print("Resize data finished!")

    if normalise:
        print('Normalising data ...')
        X = np.concatenate((not_cancer_imgs, cancer_imgs))
        X_min = X.min(axis=(0, 2, 3), keepdims=True)
        X_max = X.max(axis=(0, 2, 3), keepdims=True)
        not_cancer_imgs = (not_cancer_imgs - X_min) / (X_max - X_min)
        cancer_imgs = (cancer_imgs - X_min) / (X_max - X_min)

    # generate labels: cancer=1; no_cancer=0
    cancer_targets = np.ones((cancer_imgs.shape[0])).astype(np.int64)
    not_cancer_targets = np.zeros((not_cancer_imgs.shape[0])).astype(np.int64)
    cancer_flags = np.zeros_like(cancer_targets)

    # Generate datasets
    print("  Building datasets...")
    start = time.time()
    not_cancer_dataset = TensorDataset(torch.from_numpy(not_cancer_imgs).float(),
                                       torch.from_numpy(not_cancer_targets), torch.from_numpy(
                                           not_cancer_masks).float(),
                                       torch.from_numpy(not_cancer_flags))
    cancer_dataset = TensorDataset(torch.from_numpy(cancer_imgs).float(),
                                   torch.from_numpy(cancer_targets),
                                   torch.from_numpy(
                                       np.zeros((len(cancer_imgs), 1,
                                                 cancer_imgs.shape[2], cancer_imgs.shape[3]))).float(),
                                   torch.from_numpy(cancer_flags))

    del cancer_imgs, not_cancer_imgs, not_cancer_masks, not_cancer_flags, cancer_targets,\
        not_cancer_targets, cancer_flags
    # Build Datasets
    complete_dataset = ConcatDataset((not_cancer_dataset, cancer_dataset))

    length_complete_dataset = len(complete_dataset)

    # Build train, val and test set.
    num_total = len(complete_dataset)
    num_train = int(0.8 * num_total)
    num_test = num_total - num_train

    train_dataset, test_dataset_ = torch.utils.data.random_split(complete_dataset,
                                                                 [num_train, num_test], generator=torch.Generator().manual_seed(0))

    test_dataset_no_patches = torch.utils.data.Subset(complete_dataset,
                                                      [idx for idx in test_dataset_.indices if complete_dataset[idx][3] == 0])
    # test with not_cancer images all containing a patch
    test_dataset = torch.utils.data.Subset(complete_dataset,
                                           [idx for idx in test_dataset_.indices if complete_dataset[idx][3] == 1
                                            or complete_dataset[idx][1] == 1])

    if most_k_informative_img is not None:
        with open(informative_indices_filename, 'rb') as f:
            most_informative_ind = np.load(f)[:most_k_informative_img]
        # pytorch TensorDataset does not support assignments to update values so we have to
        # create a new TensorDataset which is equal unless the updatet flags
        imgs, labels, masks, flags = [], [], [], []
        for i, data in enumerate(train_dataset):
            imgs.append(data[0].unsqueeze(0))
            labels.append(data[1].item())
            masks.append(data[2].unsqueeze(0))
            if i in most_informative_ind and data[3] == 1:
                flags.append(1)
            else:
                flags.append(0)

        del train_dataset
        train_dataset = TensorDataset(torch.cat(imgs, 0), torch.Tensor(labels).type(torch.LongTensor),
                                      torch.cat(masks, 0), torch.Tensor(flags).type(torch.LongTensor))

        print(
            f"  MOST {most_k_informative_img} informative images with patches")
        print(
            f"  --> Train patch dist should have 1 -> {most_k_informative_img}")

    # Calculate ratio between cancerous and not_cancerous for the weighted loss in training

    cancer_ratio = len(cancer_dataset) / length_complete_dataset
    not_cancer_ratio = 1 - cancer_ratio
    cancer_weight = 1 / cancer_ratio
    not_cancer_weight = 1 / not_cancer_ratio
    weights = np.asarray([not_cancer_weight, cancer_weight])
    weights /= weights.sum()
    weights = torch.tensor(weights).float()

    datasets = {'train': train_dataset, 'test': test_dataset,
                'test_no_patches': test_dataset_no_patches}
    # tt = ConcatDataset((train_dataset, test_dataset_no_patches))

    print("  Sizes of datasets:")
    print(
        f"  TRAIN: {len(train_dataset)}, TEST: {len(test_dataset)}, TEST_NO_PATCHES: {len(test_dataset_no_patches)}")

    # only for checking the data distribution in trainset
    train_classes = [x[1].item() for x in train_dataset]
    train_patch_dis = [x[3].item() for x in train_dataset]

    # train_classes = [complete_dataset[idx][1].item() for idx in train_dataset.indices]
    # train_patch_dis = [complete_dataset[idx][3].item() for idx in train_dataset.indices]
    print(f"  TRAIN class dist: {Counter(train_classes)}")
    # 0 -> no patch, 1-> patch
    print(f"  TRAIN patch dist: {Counter(train_patch_dis)}")
    test_classes = [complete_dataset[idx][1].item()
                    for idx in test_dataset.indices]
    print(f"  TEST class dist: {Counter(test_classes)}")
    test_classes_no_patches = [
        complete_dataset[idx][1].item() for idx in test_dataset_no_patches.indices]
    print(f"  TEST_NO_PATCHES class dist: {Counter(test_classes_no_patches)}")
    print(f"  Loss weights: {str(weights)}")

    # dataloaders = {}
    # dataloaders['train'] = DataLoader(datasets['train'], batch_size=batch_size,
    #                                   shuffle=train_shuffle)
    # dataloaders['test'] = DataLoader(datasets['test'], batch_size=batch_size,
    #                                  shuffle=False)
    # dataloaders['test_no_patches'] = DataLoader(datasets['test_no_patches'], batch_size=batch_size,
    #                                             shuffle=False)

    end = time.time()
    elap = int(end - start)

    print(f"  --> Build finished: Took {elap} sec!")
    print("--------Dataset Done--------\n")
    return datasets, weights
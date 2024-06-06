import glob
import json
from pathlib import Path

from pycocotools import mask as coco_mask

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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

        self.transform = transforms.ToTensor()

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

        self.transform = transforms.ToTensor()

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

        self.transform = transforms.ToTensor()

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
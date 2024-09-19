import glob

from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image


class GlobDataset(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size
        self.total_imgs = sorted(glob.glob(root))

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        tensor_image = self.transform(image)
        return tensor_image


def get_isic_2019(datapath, number_nc=None, number_c=None, most_k_informative_img=None,
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
    # datapath = "data_store/rawdata/ISIC_2019/ISIC19/"
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
                                       np.zeros((len(cancer_imgs), 1, 299, 299))).float(),
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
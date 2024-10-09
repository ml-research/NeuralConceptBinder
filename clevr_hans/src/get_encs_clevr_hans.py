import sys
sys.path.insert(0, './clevr_hans/src/')
sys.path.insert(0, '.')
import torch
import numpy as np
import os
from tqdm import tqdm
from rtpt import RTPT

import data_clevr_hans as data
from args_clevr_hans import get_args
from neural_concept_binder import NeuralConceptBinder
import utils_clevr_hans as utils

torch.set_num_threads(40)
OMP_NUM_THREADS = 40
MKL_NUM_THREADS = 40

def transform_attrs(attrs, model):
	"""
	very sorry this is so hideous, I needed a quick hack. Here I create a one hot encoding per block, based on the
	number of concepts per block. I then conatenate them together to a tensor.
	"""
	n_blocks = attrs.shape[2]
	attrs_one_hot = [
		torch.nn.functional.one_hot(
			attrs[:, :, block_id].long(),
			num_classes=model.prior_num_concepts[block_id]
		)
		for block_id in range(n_blocks)
	]
	attrs_one_hot_cat = torch.cat(attrs_one_hot, dim=2).type(torch.FloatTensor)
	return attrs_one_hot_cat


def gather_encs(loader, model, args):
	encs = []
	encs_one_hot = []
	class_ids = []
	fnames_all = []
	# Create RTPT object
	rtpt = RTPT(name_initials='YOURINITIALS', experiment_name=f"Gather Encs",
	            max_iterations=len(loader))
	# Start the RTPT tracking
	rtpt.start()

	for i, sample in enumerate(tqdm(loader)):
		fnames = sample[3]
		imgs, _, img_class_ids = map(lambda x: x.to(args.device), sample[:3])
		out = model.encode(imgs)
		out_one_hot = transform_attrs(out[0], model)

		if args.feedback_path:
			assert out_one_hot.shape[1] == 1
			# if delete XIL is performed then remove the irrelevant concept predictions (as depicted in delete_feedback)
			# from the predicted one_hot encodings
			delete_feedback = model.cls_feedback_one_hot[img_class_ids].unsqueeze(dim=1)
			out_one_hot[delete_feedback==1] = 0.

		# for j in range(args.batch_size):
		# 	print(f'{img_class_ids[j]}: {out[0][j]}')
		# 	print(fnames[j])
		# 	print('-----------------------')

		encs.extend(out[0])
		encs_one_hot.extend(out_one_hot)
		class_ids.extend(img_class_ids)
		fnames_all.extend(fnames)
		rtpt.step()

	return torch.stack(encs_one_hot), torch.stack(encs), torch.stack(class_ids), np.array(fnames_all)


def main():
	args = get_args()
	args.model_seed = int(args.checkpoint_path.split('seed')[-1].split('/')[0])
	args.dataset = args.data_dir.split(os.path.sep)[-2]
	args.save_dir_encs = f'clevr_hans/src/tmp/{args.dataset}/'
	if args.feedback_path:
		args.save_dir_encs = f'clevr_hans/src/tmp/{args.dataset}_delete/'

	# get data
	print('Loading data ...')
	dataset_train = data.CLEVR_HANS_EXPL(
		args.data_dir, "train", lexi=True, conf_vers=args.conf_version
	)
	dataset_val = data.CLEVR_HANS_EXPL(
		args.data_dir, "val", lexi=True, conf_vers=args.conf_version
	)
	dataset_test = data.CLEVR_HANS_EXPL(
		args.data_dir, "test", lexi=True, conf_vers=args.conf_version
	)

	print(f'{len(dataset_train)}')
	print(f'{len(dataset_val)}')
	print(f'{len(dataset_test)}')

	args.n_imgclasses = dataset_train.n_classes
	args.class_weights = torch.ones(args.n_imgclasses) / args.n_imgclasses
	args.classes = np.arange(args.n_imgclasses)
	args.category_ids = dataset_train.category_ids

	train_loader = torch.utils.data.DataLoader(
		dataset_train,
		shuffle=False,
		batch_size=args.batch_size,
		pin_memory=True,
		num_workers=args.num_workers,
		drop_last=False,
	)

	val_loader = torch.utils.data.DataLoader(
		dataset_val,
		shuffle=False,
		batch_size=args.batch_size,
		pin_memory=True,
		num_workers=args.num_workers,
		drop_last=False,
	)

	test_loader = torch.utils.data.DataLoader(
		dataset_test,
		shuffle=False,
		batch_size=args.batch_size,
		pin_memory=True,
		num_workers=args.num_workers,
		drop_last=False,
	)

	print('Load model ...')
	ncb_model = NeuralConceptBinder(args)

	ncb_model.to(args.device)
	ncb_model.eval()

	if args.feedback_path:
		ncb_model.category_ids = list(np.cumsum(ncb_model.prior_num_concepts))
		ncb_model.category_ids.insert(0, 0)
		utils.set_neg_feedback(ncb_model, args)

	# create dir to save
	if not os.path.exists(args.save_dir_encs):
		os.makedirs(args.save_dir_encs)

	train_encs_one_hot, train_encs, train_labels, train_fnames = gather_encs(train_loader, ncb_model, args)
	np.save(f'{args.save_dir_encs}train_encs_one_hot_{args.model_seed}.npy', train_encs_one_hot.detach().cpu().numpy())
	np.save(f'{args.save_dir_encs}train_encs_{args.model_seed}.npy', train_encs.detach().cpu().numpy())
	np.save(f'{args.save_dir_encs}train_labels_{args.model_seed}.npy', train_labels.detach().cpu().numpy())
	np.save(f'{args.save_dir_encs}train_fnames_{args.model_seed}.npy', train_fnames)

	val_encs_one_hot, val_encs, val_labels, val_fnames = gather_encs(val_loader, ncb_model, args)
	np.save(f'{args.save_dir_encs}val_encs_one_hot_{args.model_seed}.npy', val_encs_one_hot.detach().cpu().numpy())
	np.save(f'{args.save_dir_encs}val_encs_{args.model_seed}.npy', val_encs.detach().cpu().numpy())
	np.save(f'{args.save_dir_encs}val_labels_{args.model_seed}.npy', val_labels.detach().cpu().numpy())
	np.save(f'{args.save_dir_encs}val_fnames_{args.model_seed}.npy', val_fnames)

	test_encs_one_hot, test_encs, test_labels, test_fnames = gather_encs(test_loader, ncb_model, args)
	np.save(f'{args.save_dir_encs}test_encs_one_hot_{args.model_seed}.npy', test_encs_one_hot.detach().cpu().numpy())
	np.save(f'{args.save_dir_encs}test_encs_{args.model_seed}.npy', test_encs.detach().cpu().numpy())
	np.save(f'{args.save_dir_encs}test_labels_{args.model_seed}.npy', test_labels.detach().cpu().numpy())
	np.save(f'{args.save_dir_encs}test_fnames_{args.model_seed}.npy', test_fnames)


if __name__ == "__main__":
	main()


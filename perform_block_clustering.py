import argparse
import os
import pickle
import torch
import matplotlib as mpl
import itertools
import random
import logging  # to further silence deprecation warnings
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from rtpt import RTPT
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import pairwise_distances

from data import CLEVREasyWithAnnotations, CLEVREasy_1_WithAnnotations, CLEVR4_1_WithAnnotations
from sysbinder.sysbinder import SysBinderImageAutoEncoder
import utils_ncb as utils_ncb

torch.set_num_threads(20)
OMP_NUM_THREADS = 20
MKL_NUM_THREADS = 20


THRESH_OBJ_IN_SLOT = 0.98 # found heuristically/visually

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)

parser.add_argument('--checkpoint_path', default='logs/sysbind_orig_seed0/best_model.pt')
parser.add_argument('--retrieval_corpus_path', default='logs/sysbind_orig_seed0/block_concept_dicts')

parser.add_argument('--retrieval_encs', default='prototype')
parser.add_argument('--data_path', default='data/*.png')
parser.add_argument('--log_path', default='logs/')

parser.add_argument('--lr_dvae', type=float, default=3e-4)
parser.add_argument('--lr_enc', type=float, default=1e-4)
parser.add_argument('--lr_dec', type=float, default=3e-4)
parser.add_argument('--lr_warmup_steps', type=int, default=30000)
parser.add_argument('--lr_half_life', type=int, default=250000)
parser.add_argument('--clip', type=float, default=0.05)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=4)
parser.add_argument('--num_blocks', type=int, default=8)
parser.add_argument('--cnn_hidden_size', type=int, default=512)
parser.add_argument('--slot_size', type=int, default=2048)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_prototypes', type=int, default=64)
parser.add_argument('--temp', type=float, default=1.,
					help='softmax temperature for prototype binding')
parser.add_argument('--temp_step', default=False, action='store_true')
parser.add_argument('--binarize', default=False, action='store_true',
                    help='Should the encodings of the sysbinder be binarized?')
parser.add_argument('--attention_codes', default=False, action='store_true',
                    help='Should the sysbinder prototype attention values be used as encodings?')

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
                         '(see neural_concept_binder.py for usage)')

parser.add_argument('--num_clusters', type=int, default=8)
parser.add_argument('--num_categories', type=int, default=3,
					help='how many categories of attributes')
# parser.add_argument('--clf_label_type', default='combined', choices=['combined', 'individual'],
# 					help='Specify whether the classification labels should consist of the combined attributes or '
# 						 'each attribute individually.')
parser.add_argument('--clf_type', default=None, choices=['dt', 'rg'],
					help='Specify the linear classifier model. Either decision tree (dt) or ridge regression model '
						 '(rg)')
parser.add_argument('--model_type', choices=['rb', 'sysbind'],
					help='Specify whether model type. Either original sysbinder (sysbind) or bind&retrieve (bnr).')

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_layers', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=float, default=0.1)

parser.add_argument('--tau_start', type=float, default=1.0)
parser.add_argument('--tau_final', type=float, default=0.1)
parser.add_argument('--tau_steps', type=int, default=30000)

parser.add_argument('--use_dp', default=False, action='store_true')
parser.add_argument(
		"--name",
		default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
		help="Name to store the log file as",
	)
parser.add_argument(
		"--lr", type=float, default=1e-2, help="Outer learning rate of model"
	)


def get_args():
	args = parser.parse_args()

	args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	utils_ncb.set_seed(args.seed)
	return args


def score_func(clusterer, **kwargs):
	score = clusterer.relative_validity_
	return score


# w.r.t. score_func(), greater is better or not
def compare_scores(s, t):
	return True if t > s else False


# w.r.t. compare_scores() what is the initialization score
def init_score():
	return -np.inf


def gen_param_list(X, nsteps=10):
	#     base = max(5, int(X.shape[0]/500))
	#     end = int(X.shape[0]/3)
	#     return [base + i * int(end/nsteps) for i in range(nsteps)]
	return [5, 10, 15, 20, 25, 30, 50, 80, 100]


def get_exemplars_of_cluster(cluster_id, condensed_tree):
	raw_tree = condensed_tree._raw_tree
	# Just the cluster elements of the tree, excluding singleton points
	cluster_tree = raw_tree[raw_tree['child_size'] > 1]
	# Get the leaf cluster nodes under the cluster we are considering
	leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
	# Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
	result = np.array([])
	for leaf in leaves:
		max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
		points = raw_tree['child'][(raw_tree['parent'] == leaf) &
								   (raw_tree['lambda_val'] == max_lambda)]
		result = np.hstack((result, points))
	return result.astype(np.int64)


def perform_single_param_score(p, data, scoring):
	clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True, **p)

	y_pred_test = clusterer.fit(data).labels_

	current_score = scoring(clusterer)

	return p, current_score


def custom_grid_search(X, params, scoring, verbose):
	'''
	based on: https://github.com/scikit-learn/scikit-learn/issues/17631
	'''
	data = X

	best_score = init_score()
	best_params = None

	logging.captureWarnings(True)

	# Each possible combination of parameters
	iter_params = list(itertools.product(*(params[param] for param in params)))

	# Grid search
	for i in range(len(iter_params) + 1):
		if i < len(iter_params):
			values = iter_params[i]
			p = {}
			for j, param in enumerate(params):
				p[param] = values[j]
		# perform a final test if a single cluster works best, where we set the rest to default values,
		# though they are overwritten by allow_single_clsuter: True
		elif i == len(iter_params):
			p = {"allow_single_cluster": True, "min_cluster_size": 5,
				 "min_samples": None, "cluster_selection_method": 'eom',
				 "metric": 'euclidean'}

		p, current_score = perform_single_param_score(p, data, scoring)

		if verbose:
			print(f"{p}\nscore: {current_score}")

		if compare_scores(best_score, current_score):
			best_params = p
			best_score = current_score

	return best_params, best_score


def perform_param_search(X, param_dists, reduce_dim=100, verbose=0):
	'''
	Performs randomized cv search over a specificed set of parameter distributions and returns the best set of
	parameters.
	'''

	print(f"Performing best parameter selection ... ")

	# perform dim reduction on data for quicker approximate scoring
	if reduce_dim == -1:
		X_reduced = X
	else:
		X_reduced = PCA(n_components=reduce_dim).fit_transform(X)

	# perform a grid search over the parameter distributions using the approximate validity score of hdbscan
	# package
	best_params, best_score = custom_grid_search(X=X_reduced,
												 params=param_dists,
												 scoring=score_func,
												 verbose=verbose)

	print(f"\nBest Parameters {best_params} with score {best_score}")
	return best_params


def get_all_exemplar_ids(clusterer):
	'''
	Returns the indexes of the data points that are considered 'exemplars' of each cluster, whereby the noise
	cluster is ignore. E.g if clusterer.labels_ contains ids [-1, 0, 1, 2], we receive a list of arrays, exemplar_ids for
	clusters 0, 1, and 2.
	'''

	if clusterer._prediction_data is None:
		clusterer.generate_prediction_data()

	exemplar_ids = []

	tree = clusterer.condensed_tree_
	# plt.scatter(data.T[0], data.T[1], c='grey', **plot_kwds)
	for i, c in enumerate(tree._select_clusters()):
		cluster_exemplars = get_exemplars_of_cluster(c, tree)
		exemplar_ids.append(cluster_exemplars)

	return exemplar_ids


def get_concept_dict_prototypes_and_exemplars(clusterer, blocks_single_slot_all, block_id, verbose=0):
	'''
	Given the original block representations of the slots, and the hdbscan model, identify the exemplar ids
	of the hdbscan model and compute the mean over these exemplar block representations.
	Returns a dictionary containing these averaged encodings and the corresponding cluster ids.
	As well as the exemplar encodings and the corresponding cluster ids.
	'''
	# extract the smaple ids of the exemplars (based on the hdbscan method - see api for questions)
	exemplar_ids = get_all_exemplar_ids(clusterer)

	centre_encs = []
	centre_ids = []

	# get list of cluster ids removing the noise label (-1)
	cluster_ids = np.unique(clusterer.labels_)
	cluster_ids = np.delete(cluster_ids, np.where(cluster_ids == -1)[0])

	# average over exemplar samples in the original block encoding space
	for cluster_id in cluster_ids:
		centre_encs.append(np.mean(blocks_single_slot_all[block_id, exemplar_ids[cluster_id]], axis=0))
		centre_ids.append(cluster_id)
		if verbose:
			print(f"{cluster_id} has {len(exemplar_ids[cluster_id])} exemplars")

	block_concept_dict = {
		'prototypes': {
			'prototypes': np.array(centre_encs),
			'ids': np.expand_dims(np.array(centre_ids), axis=1)
		},
		'exemplars': {
			'exemplar_ids': exemplar_ids,
			'ids': np.concatenate(
				[i * np.ones(len(cluster_exemplars)) for i, cluster_exemplars in enumerate(exemplar_ids)],
				axis=0),
			'exemplars': np.concatenate(
				[blocks_single_slot_all[block_id][exemplar_ids[i]] for i in range(len(exemplar_ids))],
				axis=0)
		},
		'params_clustering': clusterer.get_params()
	}

	return block_concept_dict


def get_concept_dict_via_exemplars(clusterer, blocks_single_slot_all, block_id, verbose=0):
	'''
	Given the original block representations of the slots, and the hdbscan model, identify the exemplar ids
	of the hdbscan model and compute the mean over these exemplar block representations.
	Returns a dictionary containing these averaged encodings and the corresponding cluster ids.
	As well as the exemplar encodings and the corresponding cluster ids.
	'''
	# extract the smaple ids of the exemplars (based on the hdbscan method - see api for questions)
	exemplar_ids = get_all_exemplar_ids(clusterer)

	centre_encs = []
	centre_ids = []

	# get list of cluster ids removing the noise label (-1)
	cluster_ids = np.unique(clusterer.labels_)
	cluster_ids = np.delete(cluster_ids, np.where(cluster_ids == -1)[0])

	# average over exemplar samples in the original block encoding space
	for cluster_id in cluster_ids:
		centre_encs.append(np.mean(blocks_single_slot_all[block_id, exemplar_ids[cluster_id]], axis=0))
		centre_ids.append(cluster_id)

		if verbose:
			print(
				f"{cluster_id} has {len(exemplar_ids[cluster_id])} exemplars")

	block_concept_dict = {
		'prototypes': {
			'prototypes': np.array(centre_encs),
			'ids': np.expand_dims(np.array(centre_ids), axis=1)
		},
		'exemplars': {
			'exemplar_ids': exemplar_ids,
			'ids': np.concatenate(
				[i * np.ones(len(cluster_exemplars)) for i, cluster_exemplars in enumerate(exemplar_ids)],
				axis=0),
			'exemplars': np.concatenate(
				[blocks_single_slot_all[block_id][exemplar_ids[i]] for i in range(len(exemplar_ids))],
				axis=0)
		},
		'params_clustering': clusterer.get_params(),
	}

	return block_concept_dict


def perform_cluster_per_block_precomputed(data, block_id, args, reduce_dim=100, n_param_steps=10, verbose=0,
										  metric='cosine'):
	'''
	Takes the data, a specific block id and perfroms a clustering via hdbscan of the data via precomputed
	distance matrix (metric specified via 'metric')
	after having performed a grid search over hdbscans parameters.
	Note: we do the parameter selection based on euclidean metric, but the final best clustering based on the
	provided metric
	'''

	assert len(data.shape) >= 3  # [nBlocks, nSamples, D], 2 for amplitude and phase
	# get data and flatten
	X = data[block_id].reshape(data.shape[1], -1)

	# specify parameters and distributions to sample from
	param_distributions = {
		"min_samples": gen_param_list(X, nsteps=n_param_steps),
		"min_cluster_size": gen_param_list(X, nsteps=n_param_steps),
		"cluster_selection_method": ["eom", "leaf"],
		"metric": ["euclidean"],
		"allow_single_cluster": [False]
	}

	verbose_params = 0
	if verbose > 1:
		verbose_params = 1

	# get best params via dbcv estimation and grid search
	best_params = perform_param_search(X, param_distributions, reduce_dim=reduce_dim, verbose=verbose_params)

	# precompute distance matrix
	distance = pairwise_distances(X, metric=metric).astype('float64')

	# perform hdbscan clustering with best params
	clusterer = hdbscan.HDBSCAN(
		min_cluster_size=best_params['min_cluster_size'],
		min_samples=best_params['min_samples'],
		metric='precomputed',
		cluster_selection_method=best_params['cluster_selection_method'],
		allow_single_cluster=best_params['allow_single_cluster']
	)
	clusterer.fit(distance)

	if verbose:
		dir = os.path.join(args.log_dir, f"clustered_examples")
		if not os.path.exists(dir):
			os.makedirs(dir)

		# visualize this clustering
		plot_dim_reduction_hdbscan(clusterer, X, method='tsne', n_components=2, noise_upper=0,
								   fp=os.path.join(dir, f'{block_id}_tsne_clustering.png')
		                           )

	return clusterer


def plot_dim_reduction_hdbscan(clusterer, X, method, n_components, verbose=0, noise_upper=5, fp=None):
	assert n_components == 2

	if len(X.shape) != 2:
		X = X.reshape(X.shape[0], -1)
	if method == 'tsne':
		X_reduced = TSNE(n_components=n_components, perplexity=50, random_state=0).fit_transform(X)
	elif method == 'pca':
		X_reduced = PCA(n_components=n_components).fit_transform(X)

	cluster_labels = clusterer.labels_
	n_samples = len(cluster_labels)

	color_palette = sns.color_palette('hls', len(np.unique(cluster_labels)))
	cluster_colors = [color_palette[x] if x >= 0
					  else (0.5, 0.5, 0.5)
					  for x in clusterer.labels_]
	cluster_member_colors = [sns.desaturate(x, p) for x, p in
							 zip(cluster_colors, clusterer.probabilities_)]
	fig, axs = plt.subplots(1, 1, figsize=(8.5, 8))
	# add random noise in case points are too close on top of another
	axs.scatter(
		X_reduced[:, 0] + np.random.uniform(-noise_upper, noise_upper, n_samples),
		X_reduced[:, 1] + np.random.uniform(-noise_upper, noise_upper, n_samples),
		s=50, linewidth=0, c=cluster_member_colors, alpha=.8
	)
	color_palette[0] = (0.5, 0.5, 0.5)

	if verbose:
		for sample_idx in range(len(clusterer.labels_)):
			axs.text(X_reduced[sample_idx, 0] + 0.01, X_reduced[sample_idx, 1] + 0.01, sample_idx)

	axs.set_title(f"{method} Embedding")

	divider = make_axes_locatable(axs)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cmap = LinearSegmentedColormap.from_list('cat_colors', color_palette, N=len(color_palette))
	fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1, len(color_palette)), cmap=cmap),
				 cax=cax, orientation='vertical', label='cluster ids')
	if fp is not None:
		try:
			fig.savefig(fp)
		except:
			pass


def plot_np_imgs(np_imgs, title='', save_fp=None):
	fig, axs = plt.subplots(5, 5)
	axs = axs.flatten()
	for i in range(len(axs)):
		if i < len(np_imgs):
			axs[i].imshow(np_imgs[i])
		axs[i].axis('off')
	fig.suptitle(title)
	fig.savefig(save_fp, bbox_inches='tight')


def plot_exemplars(block_concept_dict, imgs_all, block_id, args):
	for cluster_id in range(len(np.unique(block_concept_dict['exemplars']['ids']))):
		exemplar_dir = os.path.join(args.log_dir, f"clustered_exemplars")
		if not os.path.exists(exemplar_dir):
			os.makedirs(exemplar_dir)

		# plot_np_imgs(imgs_all[block_concept_dict['exemplars']['exemplar_ids'][cluster_id]],
		# 			 title=f"Block: {block_id} Cluster: {cluster_id}",
		# 			 save_fp=os.path.join(exemplar_dir, f"block{block_id}_{cluster_id}.png")
		#              )
		plot_np_imgs(imgs_all[block_concept_dict['exemplars']['exemplar_ids'][cluster_id]],
					 title=f"",
					 save_fp=os.path.join(exemplar_dir, f"block{block_id}_{cluster_id}.png")
		             )


def find_slot_id_with_obj(attns, args):
	'''
	This function returns the index of the slot that most likely contains the object.
	Important: we herefore assume that in fact only one object is present in the image such that we can
	filter the slot attention masks by finding that slot which contains the most attention values above a
	heuristically set threshold. This slot most likely contain the one object of the image.

	in:
	attns: [batch_size, n_slots, 1, w_img, h_img], numpy array, attention masks for each slot.
			These attention values should be between 0 and 1.
	out:
	obj_slot_ids: [batchsize] ints, between 0 and args.num_slots, indicates which slot id contains the object slot
	'''
	assert np.max(attns) <= 1. and np.min(attns) >= 0.
	assert type(attns) is np.ndarray
	# counts = [np.sum(attns[i] >= THRESH_OBJ_IN_SLOT) for i in range(args.num_slots)]
	counts = np.sum((attns >= THRESH_OBJ_IN_SLOT).reshape(args.batch_size, args.num_slots, -1), axis=2)
	obj_slot_ids = np.argmax(counts, axis=1)
	return obj_slot_ids


def slots_to_blocks(slots, args):
	"""Reshape the slot encodings to block encodings."""
	assert args.slot_size/args.num_blocks == int(args.slot_size/args.num_blocks)
	return torch.reshape(slots, (args.batch_size, args.num_slots, args.num_blocks, int(args.slot_size/args.num_blocks)))


def gather_obj_encs(model, loader, args):
	"""Iterate over all samples in loader and gather the block encodings."""
	model.eval()

	torch.set_grad_enabled(True)

	all_encs = []
	all_imgs = []
	all_img_locs = []
	for i, sample in tqdm(enumerate(loader)):
		img_locs = sample[-1]
		sample = sample[:-1]
		imgs, _, _, _ = map(lambda x: x.to(args.device), sample)

		# encode image
		slot_encs, _, attns, _ = model.encode(imgs)

		block_encs = slots_to_blocks(slot_encs, args)

		obj_slot_ids = find_slot_id_with_obj(attns.detach().cpu().numpy(), args)

		all_encs.append(block_encs[range(args.batch_size), obj_slot_ids].detach().cpu().numpy())
		all_imgs.append(np.transpose(imgs.detach().cpu().numpy(), (0, 2, 3, 1)))
		all_img_locs.extend(img_locs)

	all_encs = np.concatenate(all_encs, axis=0)
	all_imgs = np.concatenate(all_imgs, axis=0)

	return all_encs, all_imgs, all_img_locs


def main():
	args = get_args()

	assert os.path.isfile(args.checkpoint_path)

	if 'CLEVR-Easy' in args.data_path:
		train_dataset = CLEVREasy_1_WithAnnotations(
			root=args.data_path, phase="train", img_size=args.image_size, max_num_objs=args.num_slots,
			num_categories=args.num_categories,
		)
	# test_dataset = CLEVREasyWithAnnotations(
	# 	root=args.data_path, phase="test", img_size=args.image_size, max_num_objs=args.num_slots,
	# 	num_categories=args.num_categories,
	# )
	elif 'CLEVR-4' in args.data_path:
		train_dataset = CLEVR4_1_WithAnnotations(
			root=args.data_path, phase="train", img_size=args.image_size, max_num_objs=args.num_slots,
			num_categories=args.num_categories,
		)

	def seed_worker(worker_id):
		worker_seed = torch.initial_seed()
		np.random.seed(worker_seed)
		random.seed(worker_seed)

	g = torch.Generator()
	g.manual_seed(0)


	loader_kwargs = {
		"batch_size": args.batch_size,
		"shuffle": True,
		"num_workers": args.num_workers,
		"pin_memory": True,
		"drop_last": True,
		"worker_init_fn": seed_worker,
		"generator": g,
	}
	train_loader = DataLoader(train_dataset, **loader_kwargs)
	# test_loader = DataLoader(test_dataset, **loader_kwargs)

	print("-------------------------------------------\n")
	print(f"{len(train_dataset)} samples")
	print(f"{args.checkpoint_path} loading for {args.model_type} encoding classification")

	model = SysBinderImageAutoEncoder(args)
	checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
	try:
		model.load_state_dict(checkpoint['model'])
	except:
		try:
			model.load_state_dict(checkpoint)
		except:
			print('model checkpoint not in right format?')
			exit()
	args.log_dir = os.path.join(*args.checkpoint_path.split(os.path.sep)[:-1])
	print(f'loaded ...{args.checkpoint_path}')

	model.to(args.device)

	# Create and start RTPT object
	rtpt = RTPT(name_initials='YOURINITIALS', experiment_name=f"SysBinderRetriever",
				max_iterations=1)
	rtpt.start()

	# gather encodings and corresponding labels
	train_block_encs, train_imgs_np, train_img_locs = gather_obj_encs(model, train_loader, args)

	# swap batch with block axis
	train_block_encs = np.moveaxis(train_block_encs, (0, 1, 2), (1, 0, 2))

	block_concept_dicts = []
	for block_id in range(args.num_blocks):
		print(f'\nPerforming clustering of block {block_id}...')

		# perform grid search over clustering params and fit best hdbscan model
		clusterer = perform_cluster_per_block_precomputed(train_block_encs, block_id=block_id, args=args, reduce_dim=-1,
														  n_param_steps=0, verbose=1, metric='cosine')

		# get the concept dictionary for the current block based on initial block representation and exemplar ids
		block_concept_dict = get_concept_dict_via_exemplars(clusterer, train_block_encs,
		                                                                   block_id, verbose=1)
		block_concept_dicts.append(block_concept_dict)

	with open(os.path.join(args.log_dir, "block_concept_dicts.pkl"), "wb") as fp:  # Pickling
		pickle.dump(block_concept_dicts, fp)

	# plot exemplars
	for block_id in range(args.num_blocks):
		plot_exemplars(block_concept_dicts[block_id], train_imgs_np, block_id, args)

	# save all_img_locs as pikl file
	with open(os.path.join(args.log_dir, "all_img_locs.pkl"), "wb") as fp:  # Pickling
		pickle.dump(train_img_locs, fp)


if __name__ == "__main__":
	main()

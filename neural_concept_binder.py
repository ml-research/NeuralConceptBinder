from sysbinder.sysbinder import *
import pickle
import torch
import torch.nn as nn
import numpy as np


class NeuralConceptBinder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_slots = args.num_slots
        self.device = args.device
        self.slot_size = args.slot_size
        self.num_blocks = args.num_blocks
        self.softmax = nn.Softmax(dim=1)
        self.softmax_temp = 0.01
        self.majority_vote = args.majority_vote
        self.topk = args.topk

        # load retrieval corpus
        self.retrieval_corpus = self.get_retrieval_corpus(args)
        self.retrieval_encs_dim = self.retrieval_corpus[0]["encs"].shape[1]
        self.prior_num_concepts = [len(torch.unique(self.retrieval_corpus[block_id]['ids']))
                        for block_id in range(self.num_blocks)]
        print(f'\nNumber of concepts per block: \n{self.prior_num_concepts}\n')

        # integrate feedback to retriever if given:
        revision = False
        if args.deletion_dict_path:
            revision = True
            print('\n----------------------------------')
            print('Integrating deletion feedback!\n')
            self.integrate_deletion_feedback(args)
            print('----------------------------------\n')
        if args.merge_dict_path:
            revision = True
            print('\n----------------------------------')
            print('Integrating merge feedback!\n')
            self.integrate_merge_feedback(args)
            print('----------------------------------\n')

        if revision:
            self.revise_num_concepts = [len(torch.unique(self.retrieval_corpus[block_id]['ids']))
                            for block_id in range(self.num_blocks)]
            print(f'\nNumber of concepts per block with revision: \n{self.revise_num_concepts}\n')

        # load model for image encoding
        # first make sure original model passes all slots non-binarized
        args.binarize = False
        self.model = SysBinderImageAutoEncoder(args)
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

            try:
                self.model.load_state_dict(checkpoint)
            # unless a later version was used
            except:
                self.model.load_state_dict(checkpoint["model"])
                self.model.image_encoder.sysbinder.prototype_memory.attn.temp = (
                    checkpoint["temp"]
                )

            print(f"loaded ...{args.checkpoint_path}")
            args.log_dir = os.path.join(*args.checkpoint_path.split(os.path.sep)[:-1])
        else:
            print("Model path for Sysbinder was not found.")
            exit()

    def get_retrieval_corpus(self, args):
        # load retrieval corpus
        print(f"Loading retrieval corpus from {args.retrieval_corpus_path} ...")
        corpus_dict = pickle.load(open(args.retrieval_corpus_path, "rb"))
        retrieval_corpus = []
        # convert numpy arrays to torch tensors
        for block_id in range(args.num_blocks):
            if args.retrieval_encs == "proto":
                retrieval_corpus.append(
                    {
                        "encs": torch.from_numpy(
                            corpus_dict[block_id]["prototypes"]["prototypes"]
                        ).to(args.device),
                        "ids": torch.from_numpy(
                            corpus_dict[block_id]["prototypes"]["ids"]
                        ).to(args.device),
                        "types": ["prototype"]
                        * len(corpus_dict[block_id]["prototypes"]["ids"]),
                    }
                )
            elif args.retrieval_encs == "exem":
                retrieval_corpus.append(
                    {
                        "encs": torch.from_numpy(
                            corpus_dict[block_id]["exemplars"]["exemplars"]
                        ).to(args.device),
                        "ids": torch.from_numpy(
                            corpus_dict[block_id]["exemplars"]["ids"]
                        ).to(args.device),
                        "types": ["exemplar"]
                        * len(corpus_dict[block_id]["exemplars"]["ids"]),
                    }
                )
            elif args.retrieval_encs == "proto-exem":
                retrieval_corpus.append(
                    {
                        "encs": torch.from_numpy(
                            np.concatenate(
                                (
                                    corpus_dict[block_id]["prototypes"]["prototypes"],
                                    corpus_dict[block_id]["exemplars"]["exemplars"],
                                ),
                                axis=0,
                            )
                        ).to(args.device),
                        "ids": torch.from_numpy(
                            np.concatenate(
                                (
                                    np.squeeze(
                                        corpus_dict[block_id]["prototypes"]["ids"],
                                        axis=1,
                                    ),
                                    corpus_dict[block_id]["exemplars"]["ids"],
                                ),
                                axis=0,
                            )
                        ).to(args.device),
                        "types": ["prototype"]
                        * len(corpus_dict[block_id]["prototypes"]["ids"])
                        + ["exemplar"] * len(corpus_dict[block_id]["exemplars"]["ids"]),
                    }
                )
            else:
                print("ERROR: Currently only use prototype encodings for retrieval ...")
                exit()
        return retrieval_corpus


    def encode(self, imgs):
        B = imgs.shape[0]
        imgs = imgs.to(self.device)

        slots, _, attns, _ = self.model.encode(imgs)

        # get the maximal cluster ids dim: [Batch, NObjs, NBlocks] --> code
        # and the probability for that id if majority_voting is set to True, otherwise this is None:
        # [Batch, NObjs, NBlocks] --> probs
        representations = torch.stack(
            [self.retrieve_discrete_representation(s) for s in slots]
        )
        codes = representations[..., 0]
        probs = representations[..., 1]

        return codes, probs

    def forward(self):
        pass


    def retrieve_discrete_representation(self, slots):
        """
        Retrieves the discrete representations for given slots
        :param slot: Slots to retrieve the discrete representation for [num_slots, slot_size]
        :return: Discrete representations of the slots [num_slots, num_blocks]
        """
        block_size = self.slot_size // self.num_blocks
        discretized_slots = []
        for slot in slots:
            # reshape slot to single blocks
            slot = slot.reshape(self.num_blocks, block_size)

            representation = [
                self.get_closest_concept(block, self.retrieval_corpus[idx])
                for idx, block in enumerate(slot)
            ]
            discretized_slots.append(representation)

        return torch.FloatTensor(discretized_slots)

    def retrieve_discrete_block_representation(self, block_enc, block_id):
        """
        Retieves the discrete representation for a single block (instead of a full slot).
        """
        representation = self.get_closest_concept(
            block_enc, self.retrieval_corpus[block_id]
        )
        return torch.FloatTensor(representation)

    def get_closest_concept(self, block, block_retrieval_corpus):
        """
        Gets the closest concept representation to a block, returns the id of this
        :param block: Block to discretize [1, block_size]
        :param retrieval_corpus: Clusters to compare the block to [num_clusters, block_size]
        :return: Cluster id closest to the block.
        """

        # old way:
        # single_block_comparison = self.softmax_dot_product(
        #     block, block_retrieval_corpus
        # )  # distance-vector for block&clusters
        single_block_comparison = self.euclidean_distance(block, block_retrieval_corpus)

        # if we do not take a majority vote over a topk set of nearest encodings, we simply take the argmax
        if not self.majority_vote:
            return (
                block_retrieval_corpus["ids"][torch.argmin(single_block_comparison)],
                1.0,
            )
        # take the topk nearest encoding ids and select the majority occuring id
        else:
            top_ids = torch.topk(
                single_block_comparison, k=self.topk, dim=0, largest=False
            )[1]
            topk_nearest_encs_by_ids = block_retrieval_corpus["ids"][top_ids]
            # get occurence of each id in topk_nearest_encs_by_ids
            ids, occurences = torch.unique(topk_nearest_encs_by_ids, return_counts=True)
            # compute the "probability" of each id as nearest
            id_probs = occurences / self.topk
            # select the most occuring id, i.e. perform majority voting
            max_id = torch.mode(topk_nearest_encs_by_ids)[0]
            # however also collect the probability of this majority voting
            max_id_prob = id_probs[ids == max_id]
            return (max_id, max_id_prob)

    def softmax_dot_product(self, enc, block_retrieval_corpus):
        """
        Softmax product as in MarioNette (Smirnov et al. 2021)
        :param enc: [1, block_size]
        :param retrieval_corpus: {[num_clusters, block_size], [num_clusters]}
        :return: similarity scores of block and clusters [num_clusters]
        """

        enc = enc.unsqueeze(dim=0)
        retrieval_encs = block_retrieval_corpus["encs"]
        norm_factor = torch.sum(
            torch.cat(
                [
                    torch.exp(
                        torch.sum(enc * retrieval_encs[i], dim=1)
                        / np.sqrt(self.retrieval_encs_dim)
                    ).unsqueeze(dim=1)
                    for i in range(retrieval_encs.shape[0])
                ],
                dim=1,
            ),
            dim=1,
        )
        sim_scores = torch.cat(
            [
                (
                    torch.exp(
                        torch.sum(enc * retrieval_encs[i], dim=1)
                        / np.sqrt(self.retrieval_encs_dim)
                    )
                    / norm_factor
                ).unsqueeze(dim=1)
                for i in range(retrieval_encs.shape[0])
            ],
            dim=1,
        )

        # apply extra softmax to possibly enforce one-hot encoding
        sim_scores = self.softmax((1.0 / self.softmax_temp) * sim_scores)

        return sim_scores

    def euclidean_distance(self, enc, block_retrieval_corpus):
        """
        Compute the euclidean distance between the block encoding and all prototypes/exemplars.
        """
        retrieval_encs = block_retrieval_corpus["encs"]
        distances = [
            torch.linalg.vector_norm(retrieval_encs[i] - enc)
            for i in range(retrieval_encs.shape[0])
        ]
        distances = torch.tensor(distances)

        return distances

    def get_dissimilar_concepts(
        self,
        block_encoding: torch.tensor,
        id_disimilar: int,
        block_retrieval_corpus: dict,
    ):
        """
        Order concepts by disimilarity to the block encoding (using their prototypes for efficiency).
        Return the concept id of the id_dissimilar concept (1 is the concept which the block belongs to).
        """
        # compare the block encoding to all prototypes, store the similarity
        # order the similarity while keeping track of the concept id
        # return the id of the selected disimilarity

        prototypes = []

        for i, t in enumerate(block_retrieval_corpus["types"]):
            if t == "prototype":
                prototypes.append(block_retrieval_corpus["encs"][i])

        distances = torch.zeros(len(prototypes), device=prototypes[0].device)

        for i, p in enumerate(prototypes):
            distances[i] = torch.sum(block_encoding * p) / np.sqrt(
                self.retrieval_encs_dim
            )

        index = torch.topk(distances, k=id_disimilar).indices[-1]

        return index


    def delete_from_corpus(self, block_del_id, concept_del_ids):
        for concept_del_id in concept_del_ids:
            rel_ids = torch.where(self.retrieval_corpus[block_del_id]['ids'] != concept_del_id)[0]
            self.retrieval_corpus[block_del_id]['ids'] = \
                self.retrieval_corpus[block_del_id]['ids'][rel_ids]
            self.retrieval_corpus[block_del_id]['encs'] = \
                self.retrieval_corpus[block_del_id]['encs'][rel_ids]
            self.retrieval_corpus[block_del_id]['types'] = \
                [self.retrieval_corpus[block_del_id]['types'][i] \
                 for i in range(len(self.retrieval_corpus[block_del_id]['types'])) if i in rel_ids]


    def set_id_over_all_representations(self, block_del_id, concept_del_ids, set_id=-1):
        """
        Iterates over all representations and resets the ids of the cluster id identified as 'delete_id'.
        'set_id' is the novel id which the cluster encodings are set to.
        """

        def set_id_from_id_list(id_list, delete_id, set_id=-1):
            """
            This function takes a list of ids, a delete_id which should be deleted and a set_id, i.e., the value which the
            deleted ids will be set to instead.
            """
            # identify the ids of the concept-to-be-deleted
            rel_ids = id_list == delete_id
            # set these to -1
            id_list[rel_ids] = set_id
            return None

        for concept_del_id in concept_del_ids:
            set_id_from_id_list(self.retrieval_corpus[block_del_id]['ids'],
                                delete_id=concept_del_id,
                                set_id=set_id)


    def integrate_deletion_feedback(self, args):
        assert os.path.exists(args.deletion_dict_path)
        with open(args.deletion_dict_path, 'rb') as f:
            delete_concepts_dict = pickle.load(f)
        # for each block decide how to delete the irrelevant concepts
        for block_id in delete_concepts_dict.keys():
            n_clusters_block = len(torch.unique(self.retrieval_corpus[block_id]['ids']))
            # we now have three cases how to handle deletion
            # case 1: all clusters are to be deleted --> we set all cluster ids to 0
            # case 2: all clusters, but 1 are to be deleted --> we merge all to-delete clsuters,
            #         i.e., set all to-delete cluster ids to one of this set
            # case 3: at least two clusters should not be deleted --> we remove the cluster encodings completely of the
            #         to-delete clusters
            if delete_concepts_dict[block_id]:
                # case 1
                if len(delete_concepts_dict[block_id]) == n_clusters_block:
                    print(f'Integrating deletion feedback in Block {block_id} as case 1')
                    # set cluster id in corpus to 0 for all representations
                    self.set_id_over_all_representations(block_id, delete_concepts_dict[block_id], set_id=0)
                # case 2
                elif len(delete_concepts_dict[block_id]) == (n_clusters_block - 1):
                    print(f'Integrating deletion feedback in Block {block_id} as case 2')
                    # set all to-delete cluster ids to that of first one to delete, essentially merging these
                    set_id = delete_concepts_dict[block_id][0]
                    self.set_id_over_all_representations(block_id, delete_concepts_dict[block_id], set_id=set_id)
                # case 3
                elif len(delete_concepts_dict[block_id]) <= (n_clusters_block - 2):
                    print(f'Integrating deletion feedback in Block {block_id} as case 3')
                    self.delete_from_corpus(block_id, delete_concepts_dict[block_id])


    def integrate_merge_feedback(self, args):
        assert os.path.exists(args.merge_dict_path)
        with open(args.merge_dict_path, 'rb') as f:
            merge_concepts_dict = pickle.load(f)
        # for each block check if any concepts should be merged
        for block_id in merge_concepts_dict.keys():
            if merge_concepts_dict[block_id]:
                for concept_id in merge_concepts_dict[block_id].keys():
                    for concept_id_to_merge in merge_concepts_dict[block_id][concept_id].keys():
                        # if two concepts should be merged according to the feedback dictionary, then set all of the
                        # occurences of concept_id_to_merge to concept_id
                        if merge_concepts_dict[block_id][concept_id][concept_id_to_merge]:
                            print(f'Integrating merging feedback in Block {block_id}.')
                            self.set_id_over_all_representations(block_id, [concept_id_to_merge], set_id=concept_id)

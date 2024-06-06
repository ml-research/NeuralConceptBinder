import torch
import torch.nn as nn

from dvae import dVAE
from transformer import TransformerDecoder, TransformerEncoder
from utils_sysbinder import *


class BlockPrototypeMemory(nn.Module):
    def __init__(self, num_prototypes, num_blocks, d_model, temp, binarize=False):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.d_block = self.d_model // self.num_blocks

        # block prototype memory
        self.mem_params = nn.Parameter(torch.zeros(1, num_prototypes, num_blocks, self.d_block), requires_grad=True)
        nn.init.trunc_normal_(self.mem_params)
        self.mem_proj = nn.Sequential(
            linear(self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, 4 * self.d_block),
            nn.ReLU(),
            linear(4 * self.d_block, self.d_block)
        )

        # norms
        self.norm_mem = BlockLayerNorm(d_model, num_blocks)
        self.norm_query = BlockLayerNorm(d_model, num_blocks)

        # block attention
        self.attn = BlockAttention(d_model, num_blocks, temp=temp, binarize=binarize)

    def forward(self, queries):
        '''
        queries: B, N, d_model
        return: B, N, d_model
        '''

        B, N, _ = queries.shape

        # get memories
        mem = self.mem_proj(self.mem_params)  # 1, num_prototypes, num_blocks, d_block
        mem = mem.reshape(1, self.num_prototypes, -1)  # 1, num_prototypes, d_model

        # norms
        mem = self.norm_mem(mem)  # 1, num_prototypes, d_model
        queries = self.norm_query(queries)  # B, N, d_model

        # broadcast
        mem = mem.expand(B, -1, -1)  # B, num_prototypes, d_model

        # read
        return self.attn(queries, mem, mem)  # B, N, d_model


class SysBinder(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size, num_prototypes, num_blocks,
                 epsilon=1e-8, temp=0.1, binarize=False):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = BlockLayerNorm(slot_size, num_blocks)

        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = BlockGRU(slot_size, slot_size, num_blocks)
        self.mlp = nn.Sequential(
            BlockLinear(slot_size, mlp_hidden_size, num_blocks),
            nn.ReLU(),
            BlockLinear(mlp_hidden_size, slot_size, num_blocks))
        self.prototype_memory = BlockPrototypeMemory(num_prototypes, num_blocks, slot_size,
                                                     temp=temp, binarize=binarize)


    def forward(self, inputs):
        B, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            attn_logits = torch.bmm(k, q.transpose(-1, -2))
            attn_vis = F.softmax(attn_logits, dim=-1)
            # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn_vis + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(attn.transpose(-1, -2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.view(-1, self.slot_size),
                slots_prev.view(-1, self.slot_size)
            )
            slots = slots.view(-1, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
            slots, slots_blocked, attn_factor = self.prototype_memory(slots)
        
        return slots, attn_vis, (slots_blocked, attn_factor)
    

class ImageEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.cnn = nn.Sequential(
            Conv2dBlock(args.image_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.d_model, 5, 1, 2),
        )

        self.pos = CartesianPositionalEmbedding(args.d_model, args.image_size if args.image_size == 64 else args.image_size // 2)

        self.layer_norm = nn.LayerNorm(args.d_model)

        self.mlp = nn.Sequential(
            linear(args.d_model, args.d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.d_model, args.d_model))

        self.sysbinder = SysBinder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.num_prototypes, args.num_blocks,
            temp=args.temp, binarize=args.binarize)


class ImageDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.slot_proj = BlockLinear(args.slot_size, args.d_model * args.num_blocks, args.num_blocks)

        self.block_pos = nn.Parameter(torch.zeros(1, 1, args.d_model * args.num_blocks), requires_grad=True)
        self.block_pos_proj = nn.Sequential(
            BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks),
            nn.ReLU(),
            BlockLinear(args.d_model * args.num_blocks, args.d_model * args.num_blocks, args.num_blocks)
        )

        self.block_coupler = TransformerEncoder(num_blocks=1, d_model=args.d_model, num_heads=4)

        self.dict = OneHotDictionary(args.vocab_size, args.d_model)

        self.bos = nn.Parameter(torch.Tensor(1, 1, args.d_model))
        nn.init.xavier_uniform_(self.bos)

        self.decoder_pos = LearnedPositionalEmbedding1D(1 + (args.image_size // 4) ** 2, args.d_model)

        self.tf = TransformerDecoder(
            args.num_decoder_layers, (args.image_size // 4) ** 2, args.d_model, args.num_decoder_heads, args.dropout)

        self.head = linear(args.d_model, args.vocab_size, bias=False)


class SysBinderImageAutoEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.num_iterations = args.num_iterations
        self.num_slots = args.num_slots
        self.cnn_hidden_size = args.cnn_hidden_size
        self.slot_size = args.slot_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.num_prototypes = args.num_prototypes
        self.image_channels = args.image_channels
        self.image_size = args.image_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.num_blocks = args.num_blocks
        try:
            self.binarize  = args.binarize
        except:
            self.binarize = False

        try:
            self.thresh_attn_obj_slots = args.thresh_attn_obj_slots
            self.thresh_count_obj_slots = args.thresh_count_obj_slots
        except:
            self.thresh_attn_obj_slots = 0.98
            self.thresh_count_obj_slots = -1
        assert self.thresh_count_obj_slots >= -1
        if self.thresh_count_obj_slots >= 1:
            assert self.thresh_attn_obj_slots != 0

        # dvae
        self.dvae = dVAE(args.vocab_size, args.image_channels)

        # encoder networks
        self.image_encoder = ImageEncoder(args)

        # decoder networks
        self.image_decoder = ImageDecoder(args)

    def forward(self, image, tau):
        """
        image: B, C, H, W
        tau: float
        """
        B, C, H, W = image.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)  # B, vocab_size, H_enc, W_enc
        z_soft = gumbel_softmax(z_logits, tau, False, dim=1)  # B, vocab_size, H_enc, W_enc
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()  # B, vocab_size, H_enc, W_enc
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H_enc * W_enc, vocab_size
        z_emb = self.image_decoder.dict(z_hard)  # B, H_enc * W_enc, d_model
        z_emb = torch.cat([self.image_decoder.bos.expand(B, -1, -1), z_emb], dim=1)  # B, 1 + H_enc * W_enc, d_model
        z_emb = self.image_decoder.decoder_pos(z_emb)  # B, 1 + H_enc * W_enc, d_model

        # dvae recon
        dvae_recon = self.dvae.decoder(z_soft).reshape(B, C, H, W)  # B, C, H, W
        dvae_mse = ((image - dvae_recon) ** 2).sum() / B  # 1

        # sysbinder
        emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
        emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, cnn_hidden_size
        emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, H_enc * W_enc, self.d_model)  # B, H * W, cnn_hidden_size

        slots, attns, _ = self.image_encoder.sysbinder(emb_set)  # slots: B, num_slots, slot_size
        # attns: B, num_slots, num_inputs

        attns = attns \
            .transpose(-1, -2) \
            .reshape(B, self.num_slots, 1, H_enc, W_enc) \
            .repeat_interleave(H // H_enc, dim=-2) \
            .repeat_interleave(W // W_enc, dim=-1)  # B, num_slots, 1, H, W
        attns = image.unsqueeze(1) * attns + (1. - attns)  # B, num_slots, C, H, W

        # block coupling
        slots = self.image_decoder.slot_proj(slots)  # B, num_slots, num_blocks * d_model
        slots = slots + self.image_decoder.block_pos_proj(
            self.image_decoder.block_pos)  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(B, self.num_slots, self.num_blocks, -1)  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(B, self.num_slots * self.num_blocks, -1)  # B, num_slots * num_blocks, d_model

        # decode
        pred = self.image_decoder.tf(z_emb[:, :-1], slots)  # B, H_enc * W_enc, d_model
        pred = self.image_decoder.head(pred)  # B, H_enc * W_enc, vocab_size
        cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / B  # 1

        return (dvae_recon.clamp(0., 1.),
                cross_entropy,
                dvae_mse,
                attns)

    def encode(self, image):
        """
        image: B, C, H, W
        """
        B, C, H, W = image.size()

        # sysbinder
        emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
        emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, cnn_hidden_size
        emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, H_enc * W_enc, self.d_model)  # B, H * W, cnn_hidden_size

        # slots: B, num_slots, slot_size
        # attns: B, num_slots, num_inputs
        slots, attns, (slots_blocked, attns_factor) = self.image_encoder.sysbinder(emb_set)

        attns = attns \
            .transpose(-1, -2) \
            .reshape(B, self.num_slots, 1, H_enc, W_enc) \
            .repeat_interleave(H // H_enc, dim=-2) \
            .repeat_interleave(W // W_enc, dim=-1)  # B, num_slots, 1, H, W
        attns_vis = image.unsqueeze(1) * attns + (1. - attns)  # B, num_slots, C, H, W

        # get slot ids with most likely containing objects and
        # return only the object slot representations, if this is desired
        slots, attns_vis, attns, slots_blocked, attns_factor = self.find_and_select_obj_slots(
            slots, attns_vis, attns, slots_blocked, attns_factor
        )

        return slots, attns_vis, attns, (slots_blocked, attns_factor)

    def decode(self, slots):
        """
        slots: B, N, slot_size
        """
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        # block coupling
        slots = self.image_decoder.slot_proj(slots)  # B, num_slots, num_blocks * d_model
        slots = slots + self.image_decoder.block_pos_proj(
            self.image_decoder.block_pos)  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(B, num_slots, self.num_blocks, -1)  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(B, num_slots * self.num_blocks, -1)  # B, num_slots * num_blocks, d_model

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.image_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.image_decoder.tf(
                self.image_decoder.decoder_pos(input),
                slots
            )
            z_next = F.one_hot(self.image_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.image_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer.clamp(0., 1.)

    def reconstruct_autoregressive(self, image):
        """
        image: batch_size x image_channels x H x W
        """
        B, C, H, W = image.size()
        slots, _, _, _ = self.encode(image)
        recon_transformer = self.decode(slots)
        recon_transformer = recon_transformer.reshape(B, C, H, W)

        return recon_transformer

    def find_and_select_obj_slots(self, slots, attns_vis, attns, slots_blocked, attns_factor):
        '''
        This function calls find_obj_slots and for all passed arrays selects only those slot ids that are returned from
        find_obj_slots.
        '''
        B = slots.shape[0]
        # get slot ids with most likely containing objects as boolean mask
        obj_slot_ids = self.find_obj_slots(attns)

        # return only the object slot representations
        slots = torch.stack([slots[j, obj_slot_ids[j]] for j in range(B)])
        attns_vis = torch.stack([attns_vis[j, obj_slot_ids[j]] for j in range(B)])
        attns = torch.stack([attns[j, obj_slot_ids[j]] for j in range(B)])
        slots_blocked = torch.stack([slots_blocked[j, obj_slot_ids[j]] for j in range(B)])
        attns_factor = torch.stack([attns_factor[j, obj_slot_ids[j]] for j in range(B)])
        return slots, attns_vis, attns, slots_blocked, attns_factor

    def find_obj_slots(self, attns):
        '''
            This function groups the slots that most likely contains the objects vs the background.
            If thresh_count_obj_slots = -1:
            Simply pass back a tensor of ones, i.e. we select all slots.
            If thresh_count_obj_slots = 0:
            We select that slot which contains the maximum slot attention value.
            If thresh_count_obj_slots > 1:
            We filter the slot attention masks by finding that slot which contains the most attention values above a
            heuristically set threshold. These slots most likely contain the one object of the image.
            If thresh_count_obj_slots = 1:
            We take that slot with the most values above the threshold.

            in:
            attns: [n_slots, 1, w_img, h_img], torch tensor, attention masks for each slot.
                    These attention values should be between 0 and 1.
            out:
            obj_slots: those slots that most likely contain the objects
            '''
        assert torch.max(attns) <= 1. and torch.min(attns) >= 0.
        obj_slot_ids_batch = []
        for attn in attns:
            attn = attn.flatten(start_dim=1)
            if self.thresh_count_obj_slots == -1:
                obj_slot_ids = torch.ones(self.num_slots)
            elif self.thresh_count_obj_slots == 0:
                # find the maximum attention value over all slots
                max_ids = (attn == torch.max(attn)).nonzero()
                obj_slot_ids = torch.zeros(self.num_slots)
                # via majority voting select that slot id which contains the maximal attention value the most times
                # (we do this, because the maximum value can in principle occur over several slots)
                obj_slot_ids[torch.mode(max_ids[0])[0]] = 1
            else:
                # count the number of attention values above a threshold
                counts = torch.sum(attn >= self.thresh_attn_obj_slots, dim=1)
                # 0 indicates to always take the max
                if self.thresh_count_obj_slots == 1:
                    max_id = torch.argmax(counts)
                    obj_slot_ids = torch.zeros_like(counts)
                    obj_slot_ids[max_id] = 1
                # otherwise filter based on number of high attention values
                else:
                    # tensor of indexes, between 0 and args.num_slots, indicates which slot ids contains the objects
                    obj_slot_ids = counts >= self.thresh_count_obj_slots
            obj_slot_ids_batch.append(obj_slot_ids.bool())
        return obj_slot_ids_batch


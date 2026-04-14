import copy

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1, activation="relu",
                 num_feature_levels=4, enc_n_points=4,  f_token=8):
        super().__init__()
        print("+++++++++++++++++++++++using updated deformable detr++++++++++++++++++++++++++++")
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.num_feature_level = num_feature_levels

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, f_token)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, d_model, f_token=f_token)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2) # reference point here (x, y)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def forward(self, side_output, pos_embeds):
        # prepare input for encoder
        srcs = side_output
        bs, C, H, W = srcs[-1].shape

        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # [batch_size, hi*wi, c]

            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [batch_size, hi*wi, c]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)

        src_flatten = torch.cat(src_flatten, 1)  # [bs*t, \sigma(hi*wi), c]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.ones(bs, len(srcs), 2).long().to('cuda')

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten)

        # prepare input for decoder [5, 5100, 256]
        bs, _, c = memory.shape
        memory_features = []  # 8x -> 32x
        spatial_index = 0
        for lvl in range(self.num_feature_level):
            h, w = spatial_shapes[lvl]
            memory_lvl = memory[:, spatial_index : spatial_index + h * w, :].reshape(bs, h, w, c).permute(0, 3, 1, 2).contiguous()
            memory_features.append(memory_lvl)
            spatial_index += h * w

        return memory_features


class FrameTokenLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu", n_heads=8, n_levels=4, n_points=4) -> None:
        super().__init__()

        self.reference_points = nn.Linear(d_model, 2)

        # token get info from frames
        self.token_frame_atten = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # token level communication
        self.token_self_atten = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # update frame features with token
        self.frame_token_atten = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # a linear layer to generate the final output
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, token, token_pose, src_spatial_shapes, level_start_index, valid_ratios):

        B,To,C = token.shape
        # Cross atten token with src to get info from each frame.
        ref_point = self.reference_points(token).sigmoid()
        ref_point = ref_point[:, :, None] * valid_ratios[:, None]

        token2, sampling_locations, attention_weights = self.token_frame_atten(self.with_pos_embed(token, token_pose),
                                                                               ref_point, src, src_spatial_shapes,
                                                                               level_start_index)
        token = token + self.dropout1(token2)
        token = self.norm1(token)

        # Self atten between all tokens to communicate between frames
        token = rearrange(token, "b t c -> (b t) c").unsqueeze(1)  # [b*t, 1, c]
        token_pose1 = rearrange(token_pose, "b t c -> (b t) c").unsqueeze(1)
        q = k = self.with_pos_embed(token, token_pose1)

        token2 = self.token_self_atten(q, k, token)[0]
        token = token + self.dropout2(token2)
        token = self.norm2(token)

        token = rearrange(token.squeeze(1), '(b t) c -> b t c', b=B)

        # Update the frame info with token
        q = self.with_pos_embed(src, pos)
        k = self.with_pos_embed(token, token_pose)
        src2 = self.frame_token_atten(q.transpose(0, 1), k.transpose(0, 1), token.transpose(0, 1))[0].transpose(0, 1)
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        # Linear layers for output
        src2 = self.linear2(self.dropout4(self.activation(self.linear1(src))))
        src = src + self.dropout5(src2)
        src = self.norm4(src)

        return src, token


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, f_token = 8):
        super().__init__()

        # if f_token > 0:
        self.ftoken_layers = FrameTokenLayer(d_model, d_ffn,
                 dropout, activation, n_heads, n_levels, n_points)

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
            
        self.f_token = f_token
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, valid_ratios, memory_bus=None, memory_pos = None):

        if self.f_token < 0:
            src = self.inter_frame_atten(src,pos,level_start_index)
        if self.f_token > 0:
            assert (memory_bus is not None)
            src, memory_bus = self.ftoken_layers(src, pos, memory_bus, memory_pos, spatial_shapes, level_start_index, valid_ratios)


        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points,
                              src, spatial_shapes, level_start_index)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src, memory_bus

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, d_model=256, f_token = 8):
        super().__init__()
        self.f_token = f_token
        if self.f_token > 0:
            print(f"Using {f_token} frame tokens for each frame.")
            self.memory_bus = torch.nn.Parameter(torch.randn(f_token, d_model), requires_grad=True)
            self.memory_pos = torch.nn.Parameter(torch.randn(f_token, d_model), requires_grad=True)
            nn.init.kaiming_normal_(self.memory_bus, mode="fan_out", nonlinearity="relu")
            nn.init.kaiming_normal_(self.memory_pos, mode="fan_out", nonlinearity="relu")
            
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # reshape to 1D
            # valid ratios is the valid position/all position, valid pos means not masked bu padding mask
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None):
        # layers_output = []
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        if self.f_token > 0:
            B,_,_ = output.shape
            memory_bus = self.memory_bus[None,:,:].repeat(B,1,1)
            memory_pos = self.memory_pos[None, :, :].repeat(B, 1, 1)
        
        for lvl, layer in enumerate(self.layers):
            output, memory_bus = layer(output, pos, reference_points, spatial_shapes, level_start_index,
                                       valid_ratios, memory_bus, memory_pos)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_tfp(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        num_feature_levels=3,
        enc_n_points=args.enc_n_points,
        f_token=args.f_token
        )

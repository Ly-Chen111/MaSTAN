import torch
from torch import nn
import os
from .backbone import build_backbone
from .TFP_module import build_tfp
from .segmentation_head import DGDecoder
from .criterion import SetCriterion
import copy
from einops import rearrange

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)


class MaSTAN(nn.Module):
    def __init__(self, backbone_img, backbone_flow, transformer, num_queries, num_feature_levels,
                 num_frames, args=None):
        super().__init__()
        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.feature_channels = []
        # follow deformable-detr, we use the last three stages of backbone
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone_img.strides[-3:])
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone_img.num_channels[-3:][_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        self.num_frames = num_frames
        self.backbone_image = backbone_img
        self.backbone_flow = backbone_flow

        # Build DG Decoder
        feature_channels = [self.backbone_image.num_channels[0]] + 3 * [hidden_dim]

        self.decoder = DGDecoder(feature_channels=feature_channels, conv_dim=hidden_dim, norm="GN")

    def forward(self, samples: torch.Tensor, flow_maps: torch.Tensor):
        # Backbone
        features_img, pos = self.backbone_image(samples)
        features_flow, pos_flow = self.backbone_flow(flow_maps)

        b = 1   # batch size, can get from other place
        t = pos[0].shape[0] // b
        srcs = []
        poses = []

        for l, (feat, pos_l) in enumerate(zip(features_img[-3:], pos[-3:])):
            src, mask = feat.decompose()
            src = src + features_flow[-3:][l].tensors       # fuse flow features
            src_proj_l = self.input_proj[l](src)

            srcs.append(src_proj_l)
            poses.append(pos_l)

        tfp = self.transformer(srcs, poses)

        out = {}
        normal_edge, occlusion_edge = self.decoder(tfp[::-1], features_img, features_flow)

        mask_features = torch.cat([normal_edge, occlusion_edge], dim=1)
        mask_features = rearrange(mask_features, '(b t) c h w -> b t c h w', b=b, t=t)
        out['edge_pred'] = mask_features

        return out

def build_model(args):
    device = torch.device(args.device)

    # backbone
    if 'video_swin' in args.backbone:
        from .video_swin_transformer import build_video_swin_backbone
        backbone_img = build_video_swin_backbone(args)
        backbone_flow = build_video_swin_backbone(args)
    else:
        backbone_img = build_backbone(args)
        backbone_flow = build_backbone(args)

    temporal_transformer = build_tfp(args)

    model = MaSTAN(
        backbone_img,
        backbone_flow,
        temporal_transformer,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        args=args
    )
    weight_dict = {}
    weight_dict['loss_hed'] = args.hed_loss
    weight_dict['loss_focal'] = args.focal_loss
    losses = ['edge']
    criterion = SetCriterion(
            losses=losses,
            weight_dict=weight_dict
    )
    criterion.to(device)
    return model, criterion

import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('MaSTAN inference scripts.', add_help=False)
    parser.add_argument('--lr', default=2e-6, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=['backbone.0'], type=str, nargs='+')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=8, type=int)
    parser.add_argument('--lr_drop', default=[4, 6], type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Backbone
    # ["resnet50", "resnet101"]
    # ["video_swin_t_p4w7", "video_swin_s_p4w7", "video_swin_b_p4w7"]
    parser.add_argument('--backbone', default='video_swin_t_p4w7', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='whether use checkpoint for swin/video swin backbone')
    parser.add_argument('--backbone_pretrained', default=None, type=str,
                        help="if use swin backbone and train from scratch, the path to the pretrained weights")
    parser.add_argument('--dilation', action='store_true',  # DC5
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=3, type=int, help='number of feature levels')

    # Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=5, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots, all frames share the same queries")
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--hed_loss', default=3, type=float)
    parser.add_argument('--focal_loss', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='ovisoe', help='Dataset name')
    parser.add_argument('--ovisoe_path', type=str, default='', help='OVIS-OE dataset path')
    parser.add_argument('--train_num', help='which json to load', default='0')

    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--model_iter', default='/checkpoint.pth')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)

    # model variation
    parser.add_argument('--f_token', default=8, type=int, help='using frame token in encoder')

    # test setting
    parser.add_argument('--ngpu', default=1, type=int, help='gpu number')

    return parser



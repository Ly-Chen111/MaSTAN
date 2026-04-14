import argparse
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import util.misc as utils
from models import build_MaSTAN
import torchvision.transforms as T
from dataset.transforms_video import PadOVIS
import os
import cv2
from PIL import Image
import torch.nn.functional as F
import json

import inference_opts
from tqdm import tqdm


# build transform
transform = T.Compose([
    PadOVIS(320),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    

def main(args):
    args.batch_size = 1
    print("Inference only supports for batch size = 1") 
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # save path
    output_dir = args.output_dir + str(args.train_num)
    save_path_prefix = output_dir
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    # load data
    root = Path(args.ovisoe_path)
    img_folder = os.path.join(root, "Images/train")
    flow_folder = os.path.join(root, 'Flow_map')
    meta_file = os.path.join(root, "annotations_test{}.json".format(args.train_num))
    with open(meta_file, "r") as f:
        data = (json.load(f)["videos"])
    video_list = data

    start_time = time.time()
    print('Start inference')

    process_video(args, save_path_prefix, img_folder, flow_folder, video_list)

    end_time = time.time()
    total_time = end_time - start_time

    print("Total inference time: %.4f s" % (total_time))

def process_video(args, save_path_prefix, img_folder, flow_folder, video_list):
    torch.cuda.set_device(0)
    args.device = "cuda:0"

    progress = tqdm(
        total=len(video_list),
        desc="Inference Progress",
        ncols=0
    )

    # model
    model, criterion = build_MaSTAN(args)
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params:', n_parameters)

    if args.resume:
        resume_path = args.resume + str(args.train_num) + args.model_iter
        checkpoint = torch.load(resume_path, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
    	raise ValueError('Please specify the checkpoint for inference.')

    model.eval()

    for video in video_list:
        meta = {}

        meta['video_id'] = video['id']
        meta['frames'] = sorted(video['file_names'])
        meta['video_length'] = video['length']
        video_name = meta['frames'][0].split('/', 1)[0]

        if os.path.exists(os.path.join(save_path_prefix, 'png', 'normal', video_name)):     # already predicted
            print('video {} has been inferenced, skipping'.format(video_name))
            continue
        frames = meta["frames"]

        video_len = len(frames)
        start_index = 0
        if video_len >= 20:
            end_index = 20
        else:
            end_index = video_len
        while start_index < video_len:
            imgs = []
            flows = []
            for t in range(start_index, end_index):
                frame = frames[t]
                img_path = os.path.join(img_folder, frame)
                flow_path = os.path.join(flow_folder, frame.replace('.jpg', '.png'))
                if not os.path.exists(flow_path):
                    flow_path = os.path.join(flow_folder, frames[t - 1].replace('.jpg', '.png'))
                    if not os.path.exists(flow_path):
                        print('flow map {} not exist'.format(flow_path))
                        continue
                print('{}, current processing'.format((datetime.now()).strftime("%Y-%m-%d %H:%M:%S")) + img_path +
                      '\t' + 'video length=' + str(video_len) + ' index=' + str(t))
                img = Image.open(img_path).convert('RGB')
                flow_map = Image.open(flow_path).convert('RGB')
                origin_w, origin_h = img.size
                imgs.append(transform(img))
                flows.append(transform(flow_map))

            if not os.path.exists(flow_path):
                break
            imgs = torch.stack(imgs, dim=0).to(args.device)     # [video_len, 3, h, w]
            flows = torch.stack(flows, dim=0).to(args.device)

            pred_edge = slide_inference(model, imgs, flows, (origin_h, origin_w), (320, 320), (280, 280))

            normal_edges = []
            occlusion_edges = []
            edge_list = [normal_edges, occlusion_edges]
            for c in range(2):
                channel_edge = pred_edge[:, c, :, :]    # need size [t, c, d_h, d_w]
                final_edge = channel_edge.detach().cpu().numpy()
                edge_list[c].append(final_edge)

            # save binary image
            save_path = save_path_prefix
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(end_index - start_index):
                frame_name = frames[start_index+j]

                normal_edge = normal_edges[0][j]
                normal_edge_png = normal_edge * 255
                if not os.path.exists(os.path.join(save_path, 'png', 'OB', video_name)):
                    os.makedirs(os.path.join(save_path, 'png', 'OB', video_name))
                normal_save_file = os.path.join(save_path, 'png', 'OB', frame_name.replace('jpg', 'png'))
                cv2.imwrite(normal_save_file, normal_edge_png)

                occlusion_edge = occlusion_edges[0][j]
                occlusion_edge_png = occlusion_edge * 255
                if not os.path.exists(os.path.join(save_path, 'png', 'OO', video_name)):
                    os.makedirs(os.path.join(save_path, 'png', 'OO', video_name))
                occlusion_save_file = os.path.join(save_path, 'png', 'OO', frame_name.replace('jpg', 'png'))
                cv2.imwrite(occlusion_save_file, occlusion_edge_png)


            start_index += 20
            if (end_index + 20) >= video_len:
                end_index = video_len
            else:
                end_index += 20

        progress.update(1)
    progress.close()


def slide_inference(model, imgs, flows, original_size, crop_size, stride):
    '''
    Modified from EDTER.
    inference video with slide window
    1. split img to patches
    2. add each img output with pad function
    3. count_map log inferenced location
    4. output return pred_edge
    Attention: need time sequential info, so remember input size of model is [video_length, channel, h_i, w_i] [20, 3, 320, 320]
                first on grid dim(spacial), then on video_length dim(time)
               and if img size is smaller than crop_size, the img has already been padded in transforms, so need original
                size to recover size(crop padded part) after cat and return

    Args:
        imgs: [tensor] original img, with shape[video_length, channel, h, w] [20, 3, 1080, 1920]
        flows: [tensor] flow map, with shape[video_length, channel, h, w] [20, 3, 1080, 1920]
        traget: [tensor] img size -> dont need

    Output:
        pred_edge: outputs['edge_pred'] after add on, size is same as input to remove original interpolate method

    '''
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    video_length, _, h_img, w_img = imgs.size()
    size = torch.as_tensor([int(h_crop), int(w_crop)]).to(args.device)
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    # initial final img and count mat for each video frame
    preds_list = []
    count_mat_list = []
    for i in range(video_length):
        preds = imgs[i].new_zeros((1, 2, h_img, w_img))
        count_mat = imgs[i].new_zeros((1, 1, h_img, w_img))
        preds_list.append(preds)
        count_mat_list.append(count_mat)
    # for each patch
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = utils.nested_tensor_from_videos_list(imgs[:, :, y1:y2, x1:x2].unsqueeze(0), size_divisibility=32)
            crop_flow = utils.nested_tensor_from_videos_list(flows[:, :, y1:y2, x1:x2].unsqueeze(0), size_divisibility=32)
            with torch.no_grad():
                crop_seg_logit = model(crop_img, crop_flow)

            edge_pred_patch = torch.sigmoid(crop_seg_logit['edge_pred'].squeeze(0))    # [20, 2, 320, 320]
            for i in range(video_length):
                preds_list[i] += F.pad(edge_pred_patch[i, :, :, :].unsqueeze(0),
                               (int(x1), int(preds_list[i].shape[3] - x2), int(y1),
                                int(preds_list[i].shape[2] - y2)))
                count_mat_list[i][:, :, y1:y2, x1:x2] += 1

    for i in range(video_length):
        preds_list[i] = preds_list[i] / count_mat_list[i]

    preds_output = torch.cat(preds_list, dim=0)
    preds_output = preds_output[:, :, 0:original_size[0], 0:original_size[1]]

    return preds_output

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MaSTAN inference script on OVIS-OE', parents=[inference_opts.get_args_parser()])
    args = parser.parse_args()
    torch.cuda.empty_cache()
    main(args)

"""
OVIS-OE data loader
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import dataset.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random
import scipy.io as scio

#========test import
import train_opts
import argparse
from torch.utils.data import DataLoader
import util.misc as utils


class OVISOEDataset(Dataset):
    """
    A dataset class for the OVIS-OE dataset
    """
    def __init__(self, img_folder: Path, edge_folder: Path, flow_folder: Path, ann_file: Path, transforms,
                 num_frames: int, max_skip: int, args = None):
        self.args = args
        self.img_folder = img_folder
        self.edge_folder = edge_folder
        self.flow_folder = flow_folder
        self.ann_file = ann_file         
        self._transforms = transforms
        self.num_frames = num_frames     
        self.max_skip = max_skip
        self.counter = -1
        # create video meta data
        self.prepare_metas()       

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas), ' current counter: ', self.counter)  
        print('\n')    

    def refresh_metas(self):
        self.counter += 1
        self.counter %= self.num_frames
        self.prepare_metas()
        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas), ' current counter: ', self.counter)  

    def prepare_metas(self):
        # read annotation data
        with open(str(self.ann_file), 'r') as f:
            video_annotation_info = json.load(f)['videos']
        self.videos = video_annotation_info

        self.metas = []
        for vid in self.videos:
            video_name = vid['file_names'][0].split('/')[0]
            vid_frames = sorted(vid['file_names'])
            vid_len = vid['length']
            for frame_id in range(0, vid_len, self.num_frames):     #for every obj in video, from 0 to video_frame_size, 5 times add local index to record every start frame index
                meta = {}
                meta['video_id'] = vid['id']
                meta['frames'] = vid_frames
                meta['frame_id'] = frame_id
                meta['video_length'] = vid_len
                self.metas.append(meta)

        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):

        meta = self.metas[idx]  # dict
        video_id, frames, frame_id, vid_length = \
            meta['video_id'], meta['frames'], meta['frame_id'], meta['video_length']
        vid_len = vid_length

        num_frames = self.num_frames
        # random sparse sample
        sample_indx = [frame_id]
        if num_frames != 1:
            # local sample, take 1 or 2 frames from the front and back of current frame to sample, sample index->[num_frame, front frame_id, back frame_id ]
            sample_id_before = random.randint(1, 3)
            sample_id_after = random.randint(1, 3)
            local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
            sample_indx.extend(local_indx)

            # global sampling, get global frame, append the out of range frames to sample index
            if num_frames > 3:
                all_inds = list(range(vid_len))
                global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):] # the frames out of sampled index
                global_n = num_frames - len(sample_indx)
                if len(global_inds) > global_n:
                    select_id = random.sample(range(len(global_inds)), global_n)
                    for s_id in select_id:
                        sample_indx.append(global_inds[s_id])
                elif vid_len >=global_n:  # sample long range global frames
                    select_id = random.sample(range(vid_len), global_n)
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
                else:
                    select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                    for s_id in select_id:
                        sample_indx.append(all_inds[s_id])
        sample_indx.sort()
        # read frames and masks
        imgs, labels, boxes, edge_maps, flow_maps, valid = [], [], [], [], [], []
        for j in range(num_frames):
            frame_indx = sample_indx[j]
            frame_name = frames[frame_indx]
            img_path = os.path.join(str(self.img_folder), frame_name)
            edge_path = os.path.join(str(self.edge_folder), frame_name.replace(".jpg", ".mat"))
            flow_path = os.path.join(str(self.flow_folder), frame_name.replace(".jpg", ".png"))
            if not os.path.exists(flow_path):
                frame_name = frames[frame_indx - 1]     # for last frame, theres no flow map can be generated, thus use the flow map before
                flow_path = os.path.join(str(self.flow_folder), frame_name.replace(".jpg", ".png"))
            img = Image.open(img_path).convert('RGB')
            edge = scio.loadmat(edge_path)['data']
            edge_map = np.array(edge)
            edge_map = torch.from_numpy(edge_map)
            flow_map = Image.open(flow_path).convert('RGB')
            imgs.append(img)
            edge_maps.append(edge_map)
            flow_maps.append(flow_map)

        # transform
        w, h = img.size
        edge_maps = torch.stack(edge_maps, dim=0)
        target = {
            'frames_idx': torch.tensor(sample_indx), # [T,]
            'edge_maps': edge_maps,                          # [T, 2, H, W]
            'valid': False,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
        }

        valid = False
        p = 0   # drop out probability, initial to 0 to enter first loop
        while not valid and p < 0.8:    # only when valid is true or when valid is false but drop out probability > 0.8, jump out
            transformed_imgs, transformed_flow, transformed_target = self._transforms(imgs, flow_maps, target)
            valid = transformed_target['valid']  # at least one frame contain OO edge
            p = random.random()

        transformed_imgs = torch.stack(transformed_imgs, dim=0) # [T, 3, H, W]
        transformed_flow = torch.stack(transformed_flow, dim=0) # [T, 3, H, W]

        return transformed_imgs, transformed_flow, transformed_target


def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    if image_set == 'train':
        return T.Compose([
                T.Compose([
                T.CropOVIS(320),
                T.Check()
            ]),
            normalize,
        ])


    

def build(image_set, args):
    root = Path(args.ovisoe_path)
    train_num = args.train_num
    assert root.exists(), f'provided OVIS-OE path {root} does not exist'
    PATHS = {
        "train": (root / "Images" / "train", root / "Edge_2channel_mat",
                  root / "Flow_map", root / "annotations_train{}.json".format(train_num)),
        # the path of img, edge map, flow map and annotation individually
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),  # not used actually
    }
    img_folder, edge_folder, flow_folder, ann_file = PATHS[image_set]
    dataset = OVISOEDataset(img_folder, edge_folder, flow_folder, ann_file, transforms=make_transforms(image_set),
                            num_frames=args.num_frames, max_skip=args.max_skip, args = args)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MaSTAN training script', parents=[train_opts.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir + str(args.train_num)).mkdir(parents=True, exist_ok=True)
    dataset_train = build(image_set='train', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(10)
    print_freq = 10
    num_true = 0
    num_false = 0
    for samples, flow, targets in metric_logger.log_every(data_loader_train, print_freq, header):
        print("original size:{}\t".format(targets[0]['orig_size']))
        print("cropped size:{}\t".format(targets[0]['size']))
        print("edge map size:{}\t".format(targets[0]['edge_maps'].shape))
        print("frame index:{}\n".format(targets[0]['frames_idx']))
        print("valid:{}\n".format(targets[0]['valid']))
        if targets[0]['valid'] == True:
            num_true += 1
        else:
            num_false += 1
    print('true_num:{}\n'.format(num_true))
    print('false_num:{}\n'.format(num_false))


# MaSTAN
Official implement of Motion and Spatial-Temporal Aggregation Network for Occlusion Edge Detection from Videos (MaSTAN)

## Dataset
OVIS-OE dataset is now available! You can download OVIS-OE dataset from [Google Drive](https://drive.google.com/drive/folders/1TLa4AWkcFAEfbrI_BHezfMQ9kjTtdwcG?usp=drive_link).

The structure of OVIS-OE is:
```text
OVIS-OE/
├── Images/
│   └── train/                # Images for training
├── Edge/                     # Edge annotation, each image has size [H, W],
│   │                         # pixel values: {0, 1, 2} (0: background, 1: OB edge, 2: OO edge)
├── Edge_2channel_mat.zip     # Edge labels in .mat format (⚠️: ~120G after unzipping!!!)
├── annotation_train0.json    # Train split file (fold 0)
├── annotation_train1.json    # Train split file (fold 1)
├── annotation_train2.json    # Train split file (fold 2)
├── annotation_train3.json    # Train split file (fold 3)
├── annotation_train4.json    # Train split file (fold 4)
├── annotation_test0.json     # Test split file (fold 0)
├── annotation_test1.json     # Test split file (fold 1)
├── annotation_test2.json     # Test split file (fold 2)
├── annotation_test3.json     # Test split file (fold 3)
└── annotation_test4.json     # Test split file (fold 4)
```

The flow maps are generated from RATF. You can generate flow maps by yourself or download the pre-generated maps from [here](https://drive.google.com/drive/folders/1TLa4AWkcFAEfbrI_BHezfMQ9kjTtdwcG?usp=drive_link).



## Code
The code is now available.

### Installation
please refer [installation.md](https://github.com/Ly-Chen111/MaSTAN/blob/main/installation.md).

### Data
download the OVIS-OE and unzip.

### Get started
We use Video-Swin Transformer trained on Kinetics 600 as our backbone, so you need to download its pretrain weight from [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer).

#### Train
First, change the args in train_opts.py, confirm your OVIS-OE data path and other settings.

Then, you can run main.py for training.
For example, training the Video-Swin-Base model on group 0, run the following command:
```
python main.py --epochs 8 --lr_drop 4 6 --f_token 8 --num_frames 5 --output_dir=[your/output/path] --backbone video_swin_b_p4w7 --backbone_pretrained [your/path/for/backbone_pretrain] --train_num 0 --lr 2e-6
```

#### Inference
First, change the args in inference_opts.py, confirm your OVIS-OE data path and other settings.

Then, you can run inference_ovisoe.py for inference.
For example, inference the Video-Swin-Base model on group 0, run the following command:
```
python inference_ovisoe.py --backbone video_swin_b_p4w7 --f_token 8 --resume [/your/path/to/trained_model_weight] --model_iter /checkpoint.pth --output_dir=[your/output/path] --train_num 0
```

#### Model Zoo
We also provided a pretrained model for inference, you can download the checkpoint.pth from [here](https://drive.google.com/drive/folders/1TLa4AWkcFAEfbrI_BHezfMQ9kjTtdwcG?usp=drive_link) and run the above inference command.












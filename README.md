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
coming soon!

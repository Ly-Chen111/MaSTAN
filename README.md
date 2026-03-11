# MaSTAN
Official implement of Motion and Spatial-Temporal Aggregation Network for Occlusion Edge Detection from Videos (MaSTAN)

## Dataset
OVIS-OE dataset now is available!
You can download OVIS-OE dataset from \href {此处填入 Google Drive 下载地址}{Google Drive}.

The structure of OVIS-OE is:
├── Images/
│   └── train/                # Images for training
├── Edge/                     # Edge annotation, each image has a size of [H, W], and the pixel value is {0, 1, 2}, where 0 denotes background, 1 denotes OB edge, 2 denotes OO edge
|                             
│── Edge_2channel_mat.zip     # We also provide edge label in .mat format (⚠️:It is about 120G after unzipping!!!)
├── annotation_train0.json    # Train split file
├── annotation_train1.json    
├── annotation_train2.json    
├── annotation_train3.json    
├── annotation_train4.json    
├── annotation_test0.json     # Test split file
├── annotation_test1.json     
├── annotation_test2.json     
├── annotation_test3.json     
├── annotation_test4.json

The flow maps are generated from RATF. You can generate flow maps by yourself or download the maps we have already generated \href {此处填入 flow maps 下载地址}{here}.



## Code
coming soon!

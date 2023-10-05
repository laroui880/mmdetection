# dataset settings
dataset_type = 'CocoDataset'
data_root = ""

file_client_args = dict(backend='disk')

# Path of train annotation file
train_ann_file = '/hotdata/userdata/datasets/detection/NII_CU_MAPD_RGB-IR/4-channel/annotations_rgb-ir/train/coco/NII_CU_rgb-ir.json'
train_data_prefix = '/hotdata/userdata/datasets/detection/NII_CU_MAPD_RGB-IR/4-channel/images/rgb_ir/train/'  # Prefix of train image path

# Path of val annotation file
val_ann_file = '/hotdata/userdata/datasets/detection/NII_CU_MAPD_RGB-IR/4-channel/annotations_rgb-ir/val/coco/NII_CU_rgb-ir.json'
val_data_prefix = '/hotdata/userdata/datasets/detection/NII_CU_MAPD_RGB-IR/4-channel/images/rgb_ir/val/'  # Prefix of val image path
# Path of test annotation file
test_ann_file = '/hotdata/userdata/datasets/detection/NII_CU_MAPD_RGB-IR/4-channel/annotations_rgb-ir/val/coco/NII_CU_rgb-ir.json'
test_data_prefix = '/hotdata/userdata/datasets/detection/NII_CU_MAPD_RGB-IR/4-channel/images/rgb_ir/val/'  #'  # Prefix of test image path

batch_size=4
num_workers=10
persistent_workers=True
img_scale =(2688, 1952)# (832, 1088) #(640, 480)
# mean = [100.363, 88.385, 77.99474, 118.16]
# std = [37.12337, 32.39224, 29.420309, 18.1]

mean = [82.96518237, 87.00195098, 95.68444293, 118.30158302]
std = [34.37892615, 33.42075506, 37.73696908, 13.01905323]

num_classes = 1  # Number of classes for classification
classes = ["person"]

# Pipelines
train_pipeline = [
    dict(type='LoadImageFromFile', color_type= 'unchanged', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type= 'unchanged', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataloaders
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True,
        pipeline=test_pipeline))
        
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=data_root + test_ann_file,
        data_prefix=dict(img=test_data_prefix),
        test_mode=True,
        pipeline=test_pipeline))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann_file,
    metric='bbox',
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + test_ann_file,
    outfile_prefix='results/rgb_ir_person_detection')

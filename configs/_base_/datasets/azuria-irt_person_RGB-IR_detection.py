# dataset settings
dataset_type = 'CocoDataset'
data_root = ""

file_client_args = dict(backend='disk')

# Path of train annotation file
train_ann_file = '/hotdata/userdata/datasets/detection/azuria_fall_person/annotations_rgb-ir/coco/one_class_person/train.json'
train_data_prefix = '/hotdata/userdata/datasets/detection/azuria_fall_person/images/rgb-ir/'  # Prefix of train image path

# Path of val annotation file
val_ann_file = '/hotdata/userdata/datasets/detection/azuria_fall_person/annotations_rgb-ir/coco/one_class_person/val.json'
val_data_prefix = '/hotdata/userdata/datasets/detection/azuria_fall_person/images/rgb-ir/'  # Prefix of val image path
# Path of test annotation file
test_ann_file = '/hotdata/userdata/datasets/detection/azuria_fall_person/annotations_rgb-ir/coco/one_class_person/test.json'
test_data_prefix = '/hotdata/userdata/datasets/detection/azuria_fall_person/images/rgb-ir/'  #'  # Prefix of test image path

batch_size=4
num_workers=4
persistent_workers=True
img_scale = (1120, 832)

mean = [85.83402192, 113.51539475, 110.53370182, 84.36279774]
std = [53.97877444, 44.59258538, 47.29444343, 41.44217265]

num_classes = 1  # Number of classes for classification
classes = ["Person"]

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
    outfile_prefix='results/rgb_ir_fallen_person_detection')

# dataset settings
dataset_type = 'CocoDataset'
data_root = ""

file_client_args = dict(backend='disk')

batch_size=32
num_workers=10
persistent_workers=True
img_scale = (640, 480)
mean = [142.77898509, 107.60247688, 101.2352314]
std = [53.48772932, 43.03779713, 38.43625809]

num_classes = 3  # Number of classes for classification
classes_firstDataset = ["ship"]
classes_secondDataset = ["Person", "Lying_Person"]

classes = classes_firstDataset + classes_secondDataset

mixup_max_cached_images = 10

############################# SHIP ##############################

# Path of train annotation file
# train_ann_file_firstDataset = '/data/datasets_tmp/__hotdata__/detection/ship_azuria_raw/annotations/coco/train.json'
train_ann_file_firstDataset = '/data/datasets_tmp/__hotdata__/detection/ship_azuria_raw/annotations/coco/mix_ship_id3/test.json'
train_data_prefix_firstDataset = '/data/datasets_tmp/__hotdata__/detection/ship_azuria_raw/images/'  #'  # Prefix of train image path
# Path of val annotation file
val_ann_file_firstDataset = '/data/datasets_tmp/__hotdata__/detection/ship_azuria_raw/annotations/coco/mix_ship_id3/val.json'
val_data_prefix_firstDataset = '/data/datasets_tmp/__hotdata__/detection/ship_azuria_raw/images/'  # Prefix of val image path
# Path of test annotation file
test_ann_file_firstDataset = '/data/datasets_tmp/__hotdata__/detection/ship_azuria_raw/annotations/coco/mix_ship_id3/test.json'
test_data_prefix_firstDataset = '/data/datasets_tmp/__hotdata__/detection/ship_azuria_raw/images/'  #'  # Prefix of test image path

############################# FALLEN-p PERSON ####################

# Path of train annotation file
#train_ann_file_secondDataset = '/hotdata/dataset/azuria_fall_person/annotations/coco/train.json'
train_ann_file_secondDataset = '/hotdata/dataset/azuria_fall_person/annotations/coco/Mix_3datasets/test.json'

train_data_prefix_secondDataset = '/'  # Prefix of train image path
# Path of val annotation file
val_ann_file_secondDataset = '/hotdata/dataset/azuria_fall_person/annotations/coco/Mix_3datasets/val.json'
val_data_prefix_secondDataset = '/'  # Prefix of val image path
# Path of test annotation file
test_ann_file_secondDataset = '/hotdata/dataset/azuria_fall_person/annotations/coco/Mix_3datasets/test.json'
test_data_prefix_secondDataset = '/'  #'  # Prefix of test image path


# Pipelines
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=img_scale, pad_val=114.0),
    # dict(
    #     type='RandomResize',
    #     scale=(img_scale[0] * 2, img_scale[1] * 2),
    #     ratio_range=random_resize_ratio_range,
    #     keep_ratio=True),

    dict(
        type='Resize',
        scale_factor=1.0,
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale, crop_type='absolute'),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=img_scale,
        ratio_range=(1.0, 1.0), # TODO: search the value for this ratio range (not provided in mmyolo config)
        max_cached_images=mixup_max_cached_images,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    
    dict(
        type='Resize',
        scale_factor=1.0,
        keep_ratio=True),

    dict(type='RandomCrop', crop_size=img_scale, crop_type='absolute'),

    #dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    

    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


dataset_firstDataset_train = dict(
    type=dataset_type,
    data_root=train_data_prefix_firstDataset,
    metainfo=dict(classes=classes),
    ann_file=train_ann_file_firstDataset,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    data_prefix=dict(img=train_data_prefix_firstDataset),
    pipeline=train_pipeline)

dataset_secondDataset_train = dict(
    type=dataset_type,
    data_root=train_data_prefix_secondDataset,
    metainfo=dict(classes=classes),
    ann_file=train_ann_file_secondDataset,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    data_prefix=dict(img=train_data_prefix_secondDataset),
    pipeline=train_pipeline)



# dataloaders
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[dataset_firstDataset_train, dataset_secondDataset_train], ignore_keys=['classes']))


############ VAL 

dataset_firstDataset_val = dict(
    type=dataset_type,
    data_root=val_data_prefix_firstDataset,
    metainfo=dict(classes=classes),
    ann_file=val_ann_file_firstDataset,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    data_prefix=dict(img=val_data_prefix_firstDataset),
    test_mode=True,
    pipeline=test_pipeline)

dataset_secondDataset_val = dict(
    type=dataset_type,
    data_root=val_data_prefix_secondDataset,
    metainfo=dict(classes=classes),
    ann_file=val_ann_file_secondDataset,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    data_prefix=dict(img=val_data_prefix_secondDataset),
    test_mode=True,
    pipeline=test_pipeline)

# dataloaders
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[dataset_firstDataset_val, dataset_secondDataset_val], ignore_keys=['classes']))


####### TEST #############################################################

dataset_firstDataset_test = dict(
    type=dataset_type,
    data_root=test_data_prefix_firstDataset,
    metainfo=dict(classes=classes),
    ann_file=test_ann_file_firstDataset,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    data_prefix=dict(img=test_data_prefix_firstDataset),
    test_mode=True,
    pipeline=test_pipeline)

dataset_secondDataset_test = dict(
    type=dataset_type,
    data_root=test_data_prefix_secondDataset,
    metainfo=dict(classes=classes),
    ann_file=test_ann_file_secondDataset,
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    data_prefix=dict(img=test_data_prefix_secondDataset),
    test_mode=True,
    pipeline=test_pipeline)

# dataloaders
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='ConcatDataset', delete=True, datasets=[dataset_firstDataset_test, dataset_secondDataset_test], ignore_keys=['classes']))



# evaluators
val_evaluator = dict(
        type='CocoMetric',
        ann_file=data_root + val_ann_file_firstDataset,
        metric='bbox',
        format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + test_ann_file_firstDataset,
    outfile_prefix='results/Lying-p_person_ship_detection')

_base_ = [
    '../_base_/models/rtmdet_cspnext.py',
    '../_base_/datasets/smokeV2_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        frozen_stages=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', checkpoint='/home/sarah.laroui/workspace/bfte/mmselfsup/work_dirs/selfsup/swav_cspnext_8xb32-mcrop-2-6-coslr-200e_smk-224-96/epoch_200.pth.tar')))

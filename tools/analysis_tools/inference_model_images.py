# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from multiprocessing import Pool

import mmcv
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import ProgressBar, check_file_exist, mkdir_or_exist

from mmdet.datasets import get_loading_pipeline
from mmdet.evaluation import eval_map
from mmdet.registry import MODELS, DATASETS, RUNNERS
from mmdet.structures import DetDataSample
from mmdet.utils import replace_cfg_vals, update_data_root, get_test_pipeline_cfg
from mmdet.visualization import DetLocalVisualizer

from matplotlib import pyplot as plt

import torch
import cv2
from torchvision import transforms
import torch.nn as nn
from typing import Optional, Sequence, Union
from mmcv.transforms import Compose



ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def inference_detector(
    model: nn.Module,
    img: ImagesType,
    test_pipeline: Optional[Compose] = None):


    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(img, np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    # prepare data
    if isinstance(img, np.ndarray):
        # TODO: remove img_id.
        data_ = dict(img=img, img_id=0)
    else:
        # TODO: remove img_id.
        data_ = dict(img_path=img, img_id=0)

    # build the data pipeline
    data_ = test_pipeline(data_)

    data_['inputs'] = [data_['inputs']]
    data_['data_samples'] = [data_['data_samples']]

    # forward the model
    with torch.no_grad():
        results = model.test_step(data_)[0]

    return results

    

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        'image_path', help='directory where images to infer are')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument('--tta', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    from mmdet.apis import init_detector, inference_detector
    import mmcv
    import numpy as np
    import cv2
    import os
    import matplotlib.pyplot as plt
    from mmengine.structures import InstanceData
    from mmdet.structures import DetDataSample
    from mmdet.visualization import DetLocalVisualizer

    
    config_file = args.config  # './configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py' #(I have used SCNet for training)
    # load config
    cfg = Config.fromfile(config_file)

    checkpoint_file = args.checkpoint  # 'tutorial_exps/epoch_40.pth' #(checkpoint saved after training)

    model = init_detector(config_file, checkpoint_file, device='cuda:0') #loading the model
    
    img_path = args.image_path  # '/hotdata/userdata/datasets/detection/augmentation_test_dataset/images/'

    List_images = os.listdir((img_path))

    for img in os.listdir((img_path)) :

        print('img', img)

        #visualize the results in a new window
        im1 = cv2.imread(img_path + img)[:,:,::-1]
        if im1.shape[0] > im1.shape[1]:
            continue

        result = inference_detector(model, img_path + img)
        #result, x = inference_detector(model, img_path + img)
        # result, x = result[0], x[0]
        # x_flatten = torch.flatten(x)

        # print('x_flatten', x_flatten)

        pred_instances = result.pred_instances

        det_local_visualizer = DetLocalVisualizer()

        pred_det_data_sample = DetDataSample()
        pred_det_data_sample.pred_instances = pred_instances
        det_local_visualizer.add_datasample('image', im1, pred_det_data_sample, out_file= args.show_dir + img)



if __name__ == '__main__':
    main()

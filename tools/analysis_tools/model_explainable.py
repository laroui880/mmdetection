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

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        self.model.zero_grad()
        x.requires_grad_()
        out = self.model(x)
        return out

    def generate_cam(self, image_tensor, target_class):

        output = self.forward(image_tensor)

        one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float)
        one_hot_output[0][target_class] = 1
        output.backward(gradient=one_hot_output)

        gradients = self.gradients.detach().cpu().numpy()
        feature_maps = self.model.feature_maps.detach().cpu().numpy()

        cam_weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(feature_maps.shape[2:], dtype=np.float32)

        for i, weight in enumerate(cam_weights):
            cam += weight * feature_maps[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

    def register_hooks(model, grad_cam):

        def forward_hook(module, input, output):
            grad_cam.model.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            grad_cam.save_gradient(grad_output[0])

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                target_module = module

        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)


    def apply_grad_cam(image_path, model, grad_cam, mean, std, image_scale):

        # Load and preprocess image
        result = inference_detector(model, image_path)

        pred_instances = result.pred_instances

        print('pred_instances', pred_instances)
        
        # Register hooks for Grad-CAM
        GradCAM.register_hooks(model, grad_cam)
       
        probabilities = pred_instances.scores#torch.nn.functional.softmax(output, dim = 1)
        print('probabilities', probabilities.shape)
        top_prob, top_class = probabilities.topk(1, dim = 1)

        print('top_prob', top_prob)
        print('top_class', top_class)




        # # Generate Grad-CAM
        # cam = grad_cam.generate_cam(model_output, top_class)

        # # Load the original image and overlay Grad-CAM
        # original_image = cv2.imread(image_path)
        # original_image = cv2.resize(original_image, (224, 224))
        # cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        # cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        # overlayed_image = cv2.addWeighted(original_image, 0.5, cam_heatmap, 0.5, 0)

        # return top_class.item(), top_prob.item(), cam_heatmap, overlayed_image


    def visualize_results(image_path, grad_path, model, grad_cam, labels, mean, std, image_scale):   

        top_class, top_prob, cam_heatmap, overlayed_image = GradCAM.apply_grad_cam(image_path, model, grad_cam, mean, std, image_scale)
        class_label = labels[top_class]

        print('class_label', class_label)
        print('top_prob', top_prob)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(Image.open(image_path))
        ax[0].axis('off')
        ax[0].set_title('Original Image')

        ax[1].imshow(cam_heatmap)
        ax[1].axis('off')
        ax[1].set_title('Grad-CAM Heatmap')

        ax[2].imshow(overlayed_image)
        ax[2].axis('off')
        ax[2].set_title(f'Overlayed Image (Class: {class_label}, Prob: {top_prob:.4f})')

        fig.savefig(grad_path)






def save_image_gradcam_results(
                            model,
                            dataset, 
                            labels,
                            mean, 
                            std, 
                            image_scale,
                            out_dir):
    """Display or save image with groung truths and predictions from a
    model.

    Args:
        dataset (Dataset): A PyTorch dataset.
        results (list): Object detection or panoptic segmentation
            results from test results pkl file.
        performances (dict): A dict contains samples's indices
            in dataset and model's performance on them.
        out_dir (str, optional): The filename to write the image.
            Defaults: None.
        task (str): The task to be performed. Defaults: 'det'
    """
    mkdir_or_exist(out_dir)
    grad_cam = GradCAM(model)

    for data_info in dataset:

        inputs = data_info['inputs']

        # calc save file path
        filename = data_info['data_samples'].img_path
        name = filename.split("/")[-1]

        GradCAM.visualize_results(filename, out_dir + name, model, grad_cam, labels, mean, std, image_scale)
            

   

    

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
    
    config_file = args.config  # './configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py' #(I have used SCNet for training)
    # load config
    cfg = Config.fromfile(config_file)

    checkpoint_file = args.checkpoint  # 'tutorial_exps/epoch_40.pth' #(checkpoint saved after training)

    model = init_detector(config_file, checkpoint_file, device='cuda:0') #loading the model
    print('model', model)

    #     # build target layers
    # if args.target_layers:
    #     target_layers = [
    #         get_layer(layer, model) for layer in args.target_layers
    #     ]
    # else:
    #     target_layers = get_default_traget_layers(model, args)

    img = '/hotdata/userdata/datasets/detection/azuria_fall_person/images/rgb-cropped/1685527812.png'


    result = inference_detector(model, img)

    #visualize the results in a new window
    im1 = cv2.imread(img)[:,:,::-1]

    pred_instances = result.pred_instances

    from mmdet.visualization import DetLocalVisualizer

    det_local_visualizer = DetLocalVisualizer()

    pred_det_data_sample = DetDataSample()
    pred_det_data_sample.pred_instances = pred_instances
    det_local_visualizer.add_datasample('image', im1, pred_det_data_sample, out_file='/hotdata/userdata/sarah.laroui/workspace/mmdetection/tools/analysis_tools/results/out_file2.jpg')


    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    classes = cfg['classes']
    img_scale = cfg['img_scale']
    mean = cfg['mean']
    std = cfg['std']

    out_dir = '/hotdata/userdata/sarah.laroui/workspace/mmdetection/explainable_results/' 

    save_image_gradcam_results(
        model, dataset, classes, mean, std, img_scale, out_dir)


if __name__ == '__main__':
    main()

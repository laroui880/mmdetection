import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch    
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from mmdet.apis import init_detector, inference_detector


COLORS = np.random.uniform(0, 255, size=(80, 3))

def parse_detections(results):
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color, 
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img

def get_default_traget_layers(model):
    """get default target layers from given model, here choose nrom type layer
    as default target layer."""
    norm_layers = []
    for m in model.backbone.modules():
        if isinstance(m, (BatchNorm2d, LayerNorm, GroupNorm, BatchNorm1d)):
            norm_layers.append(m)
    if len(norm_layers) == 0:
        raise ValueError(
            '`--target-layers` is empty. Please use `--preview-model`'
            ' to check keys at first and then specify `target-layers`.')
    # if the model is CNN model or Swin model, just use the last norm
    # layer as the target-layer, if the model is ViT model, the final
    # classification is done on the class token computed in the last
    # attention block, the output will not be affected by the 14x14
    # channels in the last layer. The gradient of the output with
    # respect to them, will be 0! here use the last 3rd norm layer.
    # means the first norm of the last decoder block.
    
    target_layers = [norm_layers[-1]]
    return target_layers


image_url = "https://upload.wikimedia.org/wikipedia/commons/f/f1/Puppies_%284984818141%29.jpg"
img = np.array(Image.open(requests.get(image_url, stream=True).raw))
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img) / 255
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)

config_file = "/hotdata/userdata/sarah.laroui/workspace/mmdetection/configs/rtmdet/rtmdet_tiny_fallen-person_azuria-irt_RGB-cropped_test.py"  # args.config  # './configs/scnet/scnet_x101_64x4d_fpn_20e_coco.py' #(I have used SCNet for training)
# load config
cfg = Config.fromfile(config_file)

checkpoint_file = "/hotdata/userdata/sarah.laroui/workspace/mmdetection/workdir/finetune_fallen_person_azuria_irt_RGB-cropped/from_coco_rtmdet_tiny_syncbn_fast_4xb8-100e_person-fall_azuria/best_coco/bbox_mAP_epoch_97.pth" # args.checkpoint  # 'tutorial_exps/epoch_40.pth' #(checkpoint saved after training)

model = init_detector(config_file, checkpoint_file, device='cuda:0') #loading the model
print('model', model)

    # build target layers
# if args.target_layers:
#     target_layers = [
#         get_layer(layer, model) for layer in args.target_layers
#     ]
# else:
target_layers = get_default_traget_layers(model)

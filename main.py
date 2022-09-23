"""
Script based on:
- https://github.com/facebookresearch/detr
- https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb
- https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb
"""
import argparse
import numpy
import math
import random
import io
import itertools

from os import walk
from copy import deepcopy

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
import torchvision.transforms as T
torch.set_grad_enabled(False) # needed for inference?

from panopticapi.utils import rgb2id
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from util.model_loader import detr_resnet101, detr_resnet101_panoptic
from util.box_ops import box_cxcywh_to_xyxy, rescale_bboxes
from util.visualization import *

# Argparser
my_parser = argparse.ArgumentParser(description='DETR Object Detection Script')
my_parser.add_argument('--save_figs', default=False, type=bool,
                       help='[True/False]: Whether or not to save figures in directory ./out/')
my_parser.add_argument('--print_shapes', default=False, type=bool,
                       help='[True/False]: Whether or not to print shapes of certain tensors.')
my_parser.add_argument('--dataset', default="coco", type=str,
                       help='["coco"/"bsds"]: Which Dataset to use.')
my_parser.add_argument('--conf_threshold', default=0.9, type=float,
                       help='[Float in [0,1]]: Model confidence threshold to filter predictions.')
my_parser.add_argument('--img', default=None,
                       help='Filename of specific image to use.')
my_parser.add_argument('--refpoints', default=None, nargs="+", type=int,
                       help='Reference Points (y,x) to investigate attention weights. Point coordinates have to be supplied by multiple integers (e.g. "21 20 300 301" for two points (21, 20) and (300, 301)')
args = my_parser.parse_args()

# paths
ds_paths = {"coco": "/scratch-local/cdtemp/richard/datasets/coco2017/val2017/",
            "bsds": "/scratch-local/cdtemp/richard/datasets/bsds500/val/",
            }

cp_paths = {"detr_resnet101": "/scratch-local/cdtemp/richard/coding/model-checkpoints/detr/detr-r101-2c7b67e5.pth",
            "detr_resnet101_panoptic": "/scratch-local/cdtemp/richard/coding/model-checkpoints/detr/detr-r101-panoptic-40021d53.pth"
            }

# COCO classes ('N/A'-classes represent discrepancies between coco 2014 and 2017)
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def select_image():
    if not args.img:
        f = []
        for (dirpath, dirnames, filenames) in walk(ds_paths[args.dataset]):
            f.extend(filenames)
            break
        image = random.choice(f)
    else:
        image = args.img
    return image


def main():
    IMAGE = select_image()
    IM_NAME = IMAGE[:-4]

    print("----------------------------")
    print("Dataset: ", args.dataset)
    print("Image: ", IMAGE)
    print("----------------------------\n\n")

    # load pre-trained models from checkpoint files
    det_model = detr_resnet101(pretrained=True, checkpoints_path = cp_paths["detr_resnet101"])
    seg_model, postprocessor = detr_resnet101_panoptic(pretrained=True, return_postprocessor=True, num_classes=250, threshold=args.conf_threshold,
                                                       checkpoints_path = cp_paths["detr_resnet101_panoptic"])
    # get image
    im = Image.open(ds_paths[args.dataset] + IMAGE)

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], # ImageNet Mean
                    [0.229, 0.224, 0.225]) # ImageNet Std
    ])

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    # use hooks to extract attention weights (averaged over all heads)
    hooks = [
        det_model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        det_model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        det_model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model (hooks will be executed after forward pass)
    outputs = det_model(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    # keep only predictions above confidence threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > args.conf_threshold

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    assert len(bboxes_scaled) > 0, "No Objects Detected. Exiting for simplicity."

    # output of the CNN
    f_map = conv_features['0']

    # get the HxW shape of the feature maps of the CNN
    shape = f_map.tensors.shape[-2:]

    # reshape the self-attention to a more interpretable shape
    sattn = enc_attn_weights[0].reshape(shape + shape)

    if args.print_shapes:
        print("Encoder attention:      ", enc_attn_weights[0].shape)
        print("Feature map:            ", f_map.tensors.shape)
        print("Reshaped self-attention:", sattn.shape)

    plot_results(im, probas[keep], bboxes_scaled, IM_NAME, args, CLASSES)
    plot_dec_attn_weights_overlay(dec_attn_weights, h, w, im, bboxes_scaled, keep, probas, CLASSES)
    plot_dec_attn_weights(bboxes_scaled, keep, dec_attn_weights, h, w, im, probas, IM_NAME, args, CLASSES)
    plot_enc_attn_weights(sattn, im, img, IM_NAME, args)

    # SEGMENTATION USING A PRE-TRAINED MODEL FROM TORCHHUB ================================================================================================

    # Conversion table, because Detectron2 uses a different numbering scheme
    coco2d2 = {}
    count = 0
    for i, c in enumerate(CLASSES):
        if c != "N/A":
            coco2d2[i] = count
            count+=1

    out = seg_model(img)

    # compute the scores, excluding the "no-object" class (the last one) and keep only high confident ones
    scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    keep = scores > args.conf_threshold

    # Plot masks
    ncols = 5
    nrows = math.ceil(keep.sum().item() / ncols)
    wtitle = args.dataset + "-" + IM_NAME + "-bbox-attention-maps-for-object-masks"
    fig, axs = plt.subplots(num=wtitle, ncols=ncols, nrows=nrows, figsize=(18, 10))
    for line in axs:
        line = line if nrows > 1 else [line]
        for a in line:
            a.axis('off')
        for i, mask in enumerate(out["pred_masks"][keep]):
            ax = axs[i // ncols, i % ncols] if nrows > 1 else axs[i]
            ax.imshow(mask, cmap="cividis")
            ax.axis('off')
    fig.tight_layout()

    # optionally save to disc
    if args.save_figs:
        fig.savefig("out/" + wtitle)
        print("saved-4")

    # VISUALIZE SEGMENTATION ==============================================================================================

    # the post-processor expects as input the target size of the predictions (here the image size)
    result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

    palette = itertools.cycle(sns.color_palette())

    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()

    # Retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)

    # Color each mask individually
    panoptic_seg[:, :, :] = 0
    for id in range(panoptic_seg_id.max() + 1):
        panoptic_seg[panoptic_seg_id == id] = numpy.asarray(next(palette)) * 255
    wtitle = args.dataset + "-" + IM_NAME + "-segmentation"
    fig = plt.figure(num=wtitle, figsize=(15,15))
    plt.imshow(panoptic_seg)
    plt.axis('off')

    # optionally save to disc
    if args.save_figs:
        fig.savefig("out/" + wtitle)
        print("saved-5")

    # VISUALIZE PANOPTIC SEGMENTATION USING DETECTRON2 ========================================================================

    # Extract the segments info and the panoptic result from DETR's prediction
    segments_info = deepcopy(result["segments_info"])

    # Panoptic predictions are stored in a special format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    final_w, final_h = panoptic_seg.size

    # Convert the png into a segment id map
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
    panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

    # Convert class ids, because Detectron2 uses a different numbering of coco classes
    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    for i in range(len(segments_info)):
        c = segments_info[i]["category_id"]
        segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

    # Visualize the prediction
    v = Visualizer(numpy.array(im.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
    v._default_font_size = 20
    v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)

    wtitle = args.dataset + "-" + IM_NAME + "-panoptic-segmentation"
    fig = plt.figure(num=wtitle, figsize=(15,15))
    plt.imshow(v.get_image())
    plt.axis('off')

    # optionally save to disc
    if args.save_figs:
        fig.savefig("out/" + wtitle)
        print("saved-6")

    plt.show()

if __name__ == '__main__':
    for _ in range(100):
        main()

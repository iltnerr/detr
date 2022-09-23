import argparse
import io
import itertools
import math
import random

from PIL import Image
from blend_modes import *
from blend_modes.blending_functions import overlay
import cv2
import numpy
import scipy
from torch.nn import functional as F

import matplotlib.pyplot as plt
import seaborn as sns


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_dec_attn_weights_overlay(dec_attn_weights, h, w, im, bboxes_scaled, keep, probas, CLASSES):

    wtitle = "visualize decoder attention weights"
    fig, ax = plt.subplots(num=wtitle, figsize=(22, 7))
    palette = itertools.cycle(sns.color_palette())

    blended_img = im
    for idx, (xmin, ymin, xmax, ymax), c, p in zip(keep.nonzero(), bboxes_scaled, palette, probas[keep]):
        # get attention weights
        attn_weights_hw = dec_attn_weights[0, idx].view(h, w)

        # process image and weights for visualization
        if type(blended_img) == numpy.ndarray:
            blended_img = Image.fromarray(blended_img)

        # set color of weights to bbox color
        weight_color = [int(f*255) for f in list(c)]
        weight_color.append(255) # rgba

        # blend attention weights per class to input image
        blended_img = blend_img_and_weights(attn_weights_hw, blended_img, color=weight_color)
        ax.imshow(blended_img)

        # bbox + class label
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                        fill=False, color=c, linewidth=3))

        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

        ax.axis('off')

    fig.tight_layout()


def blend_img_and_weights(attn_weights_hw, im, color=[255, 0, 0, 255]):

    DEBUG = False

    im_np = numpy.asarray(im.convert("RGB"))

    # resize attention weights to image size and normalize to [0, 255] values
    attn_weights_np = F.interpolate(attn_weights_hw.unsqueeze(0).unsqueeze(0), size=im_np.shape[:2], mode='bilinear', align_corners=False).squeeze().cpu().detach().numpy()
    attn_weights_np = numpy.uint8(numpy.interp(attn_weights_np, (attn_weights_np.min(), attn_weights_np.max()), (0, 255)))

    # cv2 images
    weights = cv2.cvtColor(attn_weights_np, cv2.COLOR_RGB2BGRA)
    input_image = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGRA)

    # using CV2 and blend_modes package
    input_float = input_image.astype(float)
    weights_float = weights.astype(float)

    # Colorize weights
    color_image = numpy.zeros([weights.shape[0], weights.shape[1], 4], dtype=numpy.uint8)
    color[0], color[2] = color[2], color[0] # rgba to bgra
    color_image[:,:] = color # bgra
    color_image_float = color_image.astype(float)
    weights_colored_float = overlay(weights_float, color_image_float, opacity=1)

    # blend colored weights onto input image
    blended_img_float = addition(input_float, weights_colored_float, opacity=1)

    # convert types
    blended_img = blended_img_float.astype(numpy.uint8)
    input_image = input_float.astype(numpy.uint8)
    weights_colored = weights_colored_float.astype(numpy.uint8)

    if DEBUG:
        cv2.imshow("input image", input_image)
        cv2.imshow("attention weights upscale", attn_weights_np)
        cv2.imshow("colored weights", weights_colored)
        cv2.imshow('weights on input', blended_img)
        cv2.waitKey()  # Press a key to close window with the image.
        cv2.destroyAllWindows()

    return cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)


def plot_results(pil_img, prob, boxes, IM_NAME, args, CLASSES):
    wtitle = args.dataset + "-" + IM_NAME + "-bboxes"
    plt.figure(num=wtitle, figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

    # optionally save to disc
    if args.save_figs:
        plt.savefig("out/" + wtitle)
        print("saved-1")

def plot_dec_attn_weights(bboxes_scaled, keep, dec_attn_weights, h, w, im, probas, IM_NAME, args, CLASSES):
    wtitle = args.dataset + "-" + IM_NAME + "-decoder-attention"
    fig, axs = plt.subplots(num=wtitle, ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    colors = COLORS * 100

    # check for number of detected objects
    axs_transposed = axs.T if len(bboxes_scaled) > 1 else [axs.T]

    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs_transposed, bboxes_scaled):
        ax = ax_i[0]
        attn_weights_hw = dec_attn_weights[0, idx].view(h, w)
        ax.imshow(attn_weights_hw)
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = ax_i[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])

        fig.tight_layout()

    # optionally save to disc
    if args.save_figs:
        fig.savefig("out/" + wtitle)
        print("saved-2")

def plot_enc_attn_weights(sattn, im, img, IM_NAME, args):
    # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    fact = 32

    # select 4 random reference points with shape (y,x) to visualize attention weights.
    idxs = [(random.randrange(800), random.randrange(800)), (random.randrange(800), random.randrange(800)), (random.randrange(800), random.randrange(800)), (random.randrange(800), random.randrange(800))]

    # if provided, convert specific reference points from list to list of tuples
    refpoints = []
    if args.refpoints:
        for i in range(0, len(args.refpoints), 2):
            refpoints.append((args.refpoints[i], args.refpoints[i+1]))

    # replace random reference points with specific ones
    assert len(refpoints) <= 4, "Cannot provide more than 4 specific reference points."
    for idx, p in enumerate(refpoints):
        idxs[idx] = p

    COLORS_REFERENCE_POINTS = ["red", "green", "blue", "yellow"]

    wtitle = args.dataset + "-" + IM_NAME + "-encoder-attention"
    fig = plt.figure(num=wtitle, constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))

    # add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]

    # plot the self-attention for each reference point
    col_idx = 0
    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
        ax.axis('on')
        ax.set_title(f'self-attention{idx_o}')

        ax.patch.set_edgecolor(COLORS_REFERENCE_POINTS[col_idx])
        ax.patch.set_linewidth('5')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        col_idx += 1

    # add the central image, with the reference points as colored circles
    col_idx = 0
    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(im)
    for (y, x) in idxs:
        scale = im.height / img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color=COLORS_REFERENCE_POINTS[col_idx]))
        fcenter_ax.axis('off')

        col_idx += 1

    # optionally save to disc
    if args.save_figs:
        fig.savefig("out/" + wtitle)
        print("saved-3")


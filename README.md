# DETR: Object Detection &amp; Segmentation with DEtection TRansformer
See https://github.com/facebookresearch/detr

## Important Notes
**The paths for the trained weights and datasets are hard-coded into main.py and should be adapted.  
Model weights and datasets are not part of this repo and can be downloaded manually.**

## Example 1 (COCO)
`python main.py --img 000000117425.jpg --refpoints 625 265 300 900 200 650 447 209`

### Input Image
![input](example/ex1/input.jpg)

### Detections & Attention Weights
![output](example/ex1/output.png)

### Decoder Attention Weights for Detected Objects
![decoder-attention-raw](example/ex1/decoder-attention-raw.png)

### Encoder Attention Weights for Interesting Reference Points
![encoder-attention-refpoints](example/ex1/encoder-attention-refpoints.png)

### Object Masks
![bbox-attention-maps-for-object-masks](example/ex1/bbox-attention-maps-for-object-masks.png)

### Panoptic Segmentation
![panoptic-segmentation](example/ex1/panoptic-segmentation.png)

## Example 2 (BSDS)
`python main.py --dataset bsds --img 37073.jpg --refpoints 590 915 400 340 170 270 100 700`

![input](example/ex2/input.jpg)

![output](example/ex2/output.png)

![decoder-attention-raw](example/ex2/decoder-attention-raw.png)

![encoder-attention-refpoints](example/ex2/encoder-attention-refpoints.png)

![bbox-attention-maps-for-object-masks](example/ex2/bbox-attention-maps-for-object-masks.png)

![panoptic-segmentation](example/ex2/panoptic-segmentation.png)

## Example 3 (Mobile Phone)
![input](example/ex3/input.jpg)

![output](example/ex3/output.png)

![decoder-attention-raw](example/ex3/decoder-attention-raw.png)

![encoder-attention-refpoints](example/ex3/encoder-attention-refpoints.png)

![bbox-attention-maps-for-object-masks](example/ex3/bbox-attention-maps-for-object-masks.png)

![panoptic-segmentation](example/ex3/panoptic-segmentation.png)


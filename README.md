# DETR: Object Detection &amp; Segmentation with DEtection TRansformer (DETR)
See https://github.com/facebookresearch/detr

## Important Notes
The paths for the trained weights and datasets are hard-coded into main.py and should be adapted.

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

## Example 2 (Mobile Phone)
![input](example/ex2/input.jpg)

![output](example/ex2/output.png)

![decoder-attention-raw](example/ex2/decoder-attention-raw.png)

![encoder-attention-refpoints](example/ex2/encoder-attention-refpoints.png)

![bbox-attention-maps-for-object-masks](example/ex2/bbox-attention-maps-for-object-masks.png)

![panoptic-segmentation](example/ex2/panoptic-segmentation.png)

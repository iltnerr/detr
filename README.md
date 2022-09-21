# DETR: Object Detection &amp; Segmentation with DEtection TRansformer (DETR)
See https://github.com/facebookresearch/detr

## Important Notes
The paths for the trained weights and datasets are hard-coded into main.py and should be adapted.

## Example
```
python main.py --dataset bsds --img 253055.jpg --refpoints 540 320 510 270 200 250 530 500
```
### Input Image
![input](example/input.jpg)

### Processed Outputs: Detections & Strongest Attention Weights
![output](example/output.png)

### Decoder Attention Weights
![decoder-attention-raw](example/decoder-attention-raw.png)

### Encoder Attention Weights for Interesting Reference Points
![encoder-attention-refpoints](example/encoder-attention-refpoints.png)

### Object Masks Generated by Attention Maps
![bbox-attention-maps-for-object-masks](example/bbox-attention-maps-for-object-masks.png)

### Panoptic Segmentation
![panoptic-segmentation](example/panoptic-segmentation.png)

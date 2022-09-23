# DETR: Object Detection &amp; Segmentation with DEtection TRansformer (DETR)
See https://github.com/facebookresearch/detr

## Important Notes
The paths for the trained weights and datasets are hard-coded into main.py and should be adapted.

## Example
```
python main.py --img 000000117425.jpg --refpoints 625 265 300 900 200 650 447 209
```
### Input Image
![input](example/input.jpg)

### Detections & Attention Weights
![output](example/output.png)

### Decoder Attention Weights for Detected Objects
![decoder-attention-raw](example/decoder-attention-raw.png)

### Encoder Attention Weights for Interesting Reference Points
![encoder-attention-refpoints](example/encoder-attention-refpoints.png)

### Object Masks
![bbox-attention-maps-for-object-masks](example/bbox-attention-maps-for-object-masks.png)

### Panoptic Segmentation
![panoptic-segmentation](example/panoptic-segmentation.png)

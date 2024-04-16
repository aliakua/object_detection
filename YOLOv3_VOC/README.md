## YOLOv3 implemenatation
Here you can find implementation of Yolo_v3 from scratch on PyTorch.
According to paper Yolov3 architecture has:
* only convolutional layers(no pooling layers): for preventing of losing info during poolings
* 3 output stages for detection from small to large objects: 13*13, 26*26, 52*52
* anchor-based bboxes: based on k-means clustering
* classification predictions for each anchor

### Let's dive into details


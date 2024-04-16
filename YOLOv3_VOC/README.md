## YOLOv3 implemen


tation
Here you can find implementation of Yolo_v3 from scratch on PyTorch.
According to paper Yolov3 architecture has:
* only convolutional layers(**no pooling layers**): for preventing of losing info during poolings
* **3 output stages** for detection from small to large objects: **13x13, 26x26, 52x52**
* **anchor-based** bboxes: based on k-means clustering
* classification predictions **for each anchor**

### Let's dive into details
![image](https://github.com/aliakua/object_detection/assets/159763523/2b55ad19-2a62-42f6-9089-d2992a89b0d0)




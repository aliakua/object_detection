## YOLOv3 implementation

Here you can find implementation of Yolov3 from scratch on PyTorch.
According to paper Yolov3 architecture has:
* only convolutional layers(**no pooling layers**): for preventing of losing info during poolings
* **3 output stages** for detection from small to large objects: **13x13, 26x26, 52x52**
* **anchor-based** bboxes: based on k-means clustering
* classification predictions **for each anchor**

### Let's dive into details
1. Architecture 
    ![image](https://github.com/aliakua/object_detection/assets/159763523/2b55ad19-2a62-42f6-9089-d2992a89b0d0)
more general below
    ![image](https://github.com/aliakua/object_detection/assets/159763523/6576de64-2a93-4513-9ee7-66c2783deffa)

2. Anchor-based Bboxes:
   pw, ph - ahcors weights and heights
   cx, cy - top-left coordinates of cell
   tx, ty, tw, th - parameters which one model will be train to predict correct bx, by, bw, bh 
   ![image](https://github.com/aliakua/object_detection/assets/159763523/9691c91b-1c94-4098-b2ec-deee0be88404)






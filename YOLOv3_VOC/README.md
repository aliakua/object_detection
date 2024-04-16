## YOLOv3 implementation

Here you can find implementation of Yolov3 from scratch on PyTorch.
According to paper Yolov3 architecture has:
* only convolutional layers(**no pooling layers**): for preventing of losing info during poolings
* **3 output stages** for detection from small to large objects: **13x13, 26x26, 52x52**
* **anchor-based** bboxes: for eliminating unstable gradients
* classification predictions **for each anchor**

### Let's dive into details
1. Architecture
   
    ![image](https://github.com/aliakua/object_detection/assets/159763523/2b55ad19-2a62-42f6-9089-d2992a89b0d0)
more general below

    ![image](https://github.com/aliakua/object_detection/assets/159763523/e1368644-7bb8-4775-92aa-a8e4e63bda08)

    ![image](https://github.com/aliakua/object_detection/assets/159763523/6576de64-2a93-4513-9ee7-66c2783deffa)

3. Anchor-based Bboxes: based on k-means clustering
   - pw, ph - anchor widht and height
   - cx, cy - cells top-left coordinates of the anchor box
   - tx, ty, tw, th - outputs of NN, which one model will train to predict for correct bx, by, bw, bh
      
    ![image](https://github.com/aliakua/object_detection/assets/159763523/9691c91b-1c94-4098-b2ec-deee0be88404)
4. Final predictions for 3 outputs look like
   
   ![image](https://github.com/aliakua/object_detection/assets/159763523/9a8c4da2-c39c-4c6b-bf50-f3ab55d96ac1)

5. Objectness score
   * YOLOv3 predicts an objectness score for each bounding box using logistic regression
7. Loss: Regression Loss + Confidence Loss + Classification Loss
   * YOLOv3 changes the way in calculating the cost function. If the bounding box prior (anchor) overlaps a ground truth object more than others, the corresponding objectness score should be 1.
   * For other priors with overlap greater than a predefined threshold (default 0.5), they incur no cost.
   * Each ground truth object is associated with one boundary box prior only. If a bounding box prior is not assigned, it incurs no classification and localization lost, just confidence loss on objectness. We use tx and ty (instead of bx and by) to compute the loss.

   ![image](https://github.com/aliakua/object_detection/assets/159763523/b8c55366-29dc-4117-b74a-147ce9aa5019)



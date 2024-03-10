### About project
For diving deeply into state-of-the-art **YOLOv1(You-only-look-once)** this project was done accordingly https://www.youtube.com/watch?v=n9_XyCGr-MI&t=1836s.
So many thanks to author of lesson - Aladdin Persson - for nice explaining.

#### Data
For object detection was used **Pascal Voc dataset** from kaggle. 
`! kaggle datasets download -d gopalbhattrai/pascal-voc-2012-dataset`

Bcs of YOLO is grid-cell based object detetion technology, was necessary to do some preparation of data, which one you can find at prepare_data.py.   

#### Metric
As metric of model accuracy is used mAP(mean Average Precision) with fixed IOU = 0.5.

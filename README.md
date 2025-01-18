This a custom object detection model using YOLOv11(yolo11n.pt) model. It is running in the machine locally wwith nvidia cuda gpu.
The dataset(weapons) for this project is downlaoded from roboflow, which already had training, validation and testing images along with their respective annotations.
The model has been trained for 50 epochs, with batch size=8.
The model has been deployed into web app, using Flask along with html which takes inout image from the device and returns the output with predicted image, bounding boxes with their labels and confidence scores.

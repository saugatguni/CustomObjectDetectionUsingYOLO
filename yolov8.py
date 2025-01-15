from ultralytics import YOLO

model= YOLO('yolov8n.yaml') #lightweight version

results=model.train(data='data.yaml' , epochs=3)
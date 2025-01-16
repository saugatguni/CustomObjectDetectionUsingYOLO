from ultralytics import YOLO

model= YOLO('yolo11n.pt') #lightweight version

results=model.train(data='data.yaml',  
                    epochs=50,
                    batch=8,
                    imgsz=640,
                    device=0)
from flask import Flask, render_template, request, url_for
import os
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

app=Flask(__name__)
model=YOLO("runs/detect/train14/weights/best.pt")

os.makedirs("static/results", exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    
    file=request.files['image']
    if file.filename=='':
        return "No file selected", 400
    
    image=Image.open(file.stream)
    image_cv=np.array(image)
    image_cv=cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    results=model.predict(image_cv)

    for box in results[0].boxes:
        x1,y1,x2,y2=box.xyxy[0]
        label = f"{model.names[int(box.cls)]} ({box.conf[0]:.2f})"

        cv2.rectangle(image_cv,(int(x1), int(y1)), (int(x2),int(y2)),(0,255,0),3)
        cv2.putText(image_cv, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (0,0,255),2)

        return render_template('result.html', image_url=url_for('static', filename='results/results_with_boxes.jpg'))
    
if __name__=="__main__":
    app.run()
from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

app = Flask(__name__)
model = YOLO("runs/detect/train14/weights/best.pt")

# Ensure results directory exists
os.makedirs("static/results", exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Open the image
    image = Image.open(file.stream)

    # Run YOLO prediction
    results = model.predict(image)

    # Draw bounding boxes and predictions on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        label = f"{model.names[int(box.cls)]} ({box.conf[0]:.2f})"
        
        # Draw rectangle and label
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), label, fill="red", font=font)

    # Save the modified image
    image.save("static/results/result_with_boxes.jpg")

    # Return the image with bounding boxes and predictions drawn
    return render_template('result.html', image_url="/static/results/result_with_boxes.jpg")

if __name__ == "__main__":
    app.run(debug=True)

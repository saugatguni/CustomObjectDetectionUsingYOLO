from ultralytics import YOLO
import cv2

model=YOLO("runs/detect/train14/weights/best.pt")

results=model("test.jpg")

for r in results:
    # Plot the results (YOLO provides a plot method)
    annotated_image = r.plot()  # This creates an image with bounding boxes drawn

    # Display the image using OpenCV
    cv2.imshow('YOLO Detection', annotated_image)

    # Press any key to close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

from ultralytics import YOLO
from ultralytics.engine.results import Results
import os
import cv2

class MyBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = xyxy
        self.conf = conf

    def show(self):
        print(f"xyxy: {self.xyxy}")
        print(f"conf: {self.conf}")

def process_image(image_path):
    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8x-seg.pt')

    # Run inference on an image
    results: Results = model(image_path)  # results list

    boxes = list()

    # View results
    for r in results:
        for box in r.boxes:
            my_box = MyBoxes(box.xyxy, box.conf)
            boxes.append(my_box)

    return boxes, results[0].plot()  # Return boxes and the image with plots

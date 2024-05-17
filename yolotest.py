from ultralytics import YOLO
from ultralytics.engine.results import Results

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-seg.pt')

# Run inference on an image
results: Results = model('20240502_142118.jpg')  # results list

# View results
for r in results[0]:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes
    print(r.boxes.xyxy)  # print the xyxy tensor of bounding boxes
    print(r.boxes.conf)  # print the confidence tensor of bounding boxes

results[0].show()  # display the image with bounding boxes

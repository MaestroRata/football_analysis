from ultralytics import YOLO

# Choose the biggest version of the model to have better results
# if your hardware can't keep up choose one smaller
# tried it on M1 macbook air and working fine
model = YOLO("yolov8x")

# This command will run the model to detect objects in the image
# or video selected, and save the output into the 'runs/detect/predict' folder
results = model.predict("input_videos/08fd33_4.mp4", save=True)
print(results[0])
print("######################################")
for box in results[0].boxes:
    print(box)

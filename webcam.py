from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model(0,show= True)

for result in results:
    boxes= result.boxes
    classes = result.names

    if boxes > 6:
        break

print(boxes)      


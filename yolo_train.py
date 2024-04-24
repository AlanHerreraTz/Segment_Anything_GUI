from ultralytics import YOLO

model = YOLO("/home/alan/gui_sam/yolov8n.pt")

results = model.train(data= "/home/alan/gui_sam/data.yaml", epochs = 70, device = 0, patience = 15 )
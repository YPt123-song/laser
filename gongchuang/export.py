from ultralytics import YOLO

# Load a 
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("D:\gongchuang\\runs\detect\\train3\weights\\best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx",dynamic=True)
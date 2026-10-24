# Run this ONCE in Google Colab (free), not on your laptop.
# 1) Open https://colab.research.google.com -> New notebook
# 2) Paste each line below into a cell and run, then download yolov8n.onnx
#
#   !pip install -q ultralytics
#   from ultralytics import YOLO
#   YOLO("yolov8n.pt").export(format="onnx", imgsz=320, opset=12, simplify=True)
#   from google.colab import files; files.download("yolov8n.onnx")
#
# 3) Put the downloaded file at:  visionguard/models/yolov8n.onnx

# ---- OPTIONAL: more accurate 640 model (slower, about 4x per frame, still fine for offline analysis) ----
#   !pip install -q ultralytics
#   from ultralytics import YOLO
#   YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=12, simplify=True)
#   import shutil; shutil.copy("yolov8n.onnx", "yolov8n-640.onnx")
#   from google.colab import files; files.download("yolov8n-640.onnx")
# Put yolov8n-640.onnx in the models/ folder next to yolov8n.onnx. It is picked up automatically.

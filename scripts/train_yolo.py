from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", required=False, default="data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    model = YOLO('yolov8n.pt')
    model.train(data=args.data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

import os
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class YOLOBackend:
    def __init__(self, weights, conf=0.25, iou=0.45):
        if YOLO is None:
            raise RuntimeError("Ultralytics not installed")
        if not os.path.exists(weights):
            raise FileNotFoundError(weights)
        self.model = YOLO(weights); self.conf=conf; self.iou=iou
    def infer(self, image_bgr):
        r = self.model(image_bgr, conf=self.conf, iou=self.iou)[0]
        boxes=[]; labels=[]; scores=[]
        if hasattr(r,'boxes'):
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy().tolist()
            names = r.names
            for (x1,y1,x2,y2), s in zip(xyxy, confs):
                boxes.append((int(x1),int(y1),int(x2),int(y2)))
                labels.append("defect")
                scores.append(float(s))
        return boxes, labels, scores

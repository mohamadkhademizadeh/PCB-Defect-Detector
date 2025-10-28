# PCB-Defect-Detector

Classical + deep learning pipeline to detect **PCB surface defects** (scratches, opens, mouse bites, solder bridges).
- **Classical pipeline** (morphology + contour + template diff) works **out-of-the-box**.
- Optional **YOLOv8** backend (if you have labeled data).
- Streamlit app to upload PCB images and get masks & boxes.
- Training scripts for YOLO + dataset prep template.

---

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/pcb_app.py
```
Then upload a pair: a **golden board** and a **test board** or a single test board (fallback to morphology).

---

## Layout
```
PCB-Defect-Detector/
├── app/pcb_app.py
├── utils/
│   ├── classical.py        # thresholding, template match, contour filtering
│   ├── yolo_backend.py     # YOLO adapter (optional)
│   ├── viz.py              # overlays
│   └── io.py               # image I/O & alignment helper
├── scripts/
│   ├── prepare_dataset.py  # convert your data to YOLO format
│   └── train_yolo.py
├── configs/default.yaml
├── models/                 # place weights here
├── tests/test_classical.py
├── requirements.txt
└── README.md
```

**Notes:** For best results, align the board to a golden reference (feature-based alignment included).

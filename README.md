# 🛡️Military-Soldier-Safety-and-Weapon-Detection-using-YOLO-and-Computer-Vision

This project implements a deep learning pipeline for detecting military-related objects—such as camouflage soldiers, military tanks, and trucks—using YOLOv8. The system leverages annotated image data to train and evaluate high-performance object detection models and provides an interactive interface for testing on images and videos.

---

## 📁 Project Structure

```
military_detection/
│
├── data/
│   ├── images/              # Original image dataset
│   ├── labels/              # YOLO-format label files
│   └── data.yaml            # Dataset configuration file
│
├── runs/                    # YOLO training and evaluation runs
│
├── best.pt                  # Best trained model
│
├── military_streamlit.py    # Streamlit app for interactive inference
│
└── README.md                # Project documentation
```

---

## 🧠 Models and Results

| Run Name                        | Epochs | mAP50   | mAP50-95 |
|-------------------------------|--------|---------|----------|
| `military_detection_finetune` | 20     | 0.91510 | 0.68654  |
| `military_detection_full_finetune` | 15 | 0.89977 | 0.65999  |
| `military_detection_continue` | 30     | 0.83542 | 0.58832  |
| `military_detection4`         | 30     | 0.81523 | 0.56661  |

---

## 📊 Evaluation Metrics

| Class             | Precision | Recall | mAP50 | mAP50-95 |
|------------------|-----------|--------|-------|----------|
| Camouflage Soldier | 0.869     | 0.804  | 0.863 | 0.528    |
| Military Tank     | 0.819     | 0.894  | 0.920 | 0.649    |
| Military Truck    | 0.848     | 0.790  | 0.844 | 0.629    |
| **All Classes**   | 0.845     | 0.829  | 0.876 | 0.602    |

---



## 🚀 Running the Streamlit App

To start the interface:

```bash
streamlit run military_streamlit.py
```

Features:
- Upload image or video
- See real-time detection results


---

## 📈 Model Training (YOLOv8)

To train:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=30 imgsz=640
```



To evaluate:

```bash
yolo task=detect mode=val model=best.pt data=data.yaml
```

<img width="655" height="693" alt="flowchart0" src="https://github.com/user-attachments/assets/f46292bf-6dab-4237-ad8a-3bb61823f803" />


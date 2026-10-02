# 🛡️ SentinelAI — Smart Intrusion Detection System

## Implementation Plan

> **Goal**: Build a real-time intrusion detection system that uses OpenCV frame differencing to detect motion and a PyTorch CNN to classify what moved — Person, Animal, Vehicle, or False Alarm — then triggers smart alerts only for genuine threats.

---

## 📁 Project Structure

```
SentinelAI/
├── IMPLEMENTATION_PLAN.md          # This file
├── requirements.txt                # All dependencies
├── config.py                       # Centralized configuration
├── main.py                         # Entry point — runs the full pipeline
│
├── src/                            # Core source code
│   ├── __init__.py
│   ├── motion_detector.py          # Frame differencing (from your object-tracking.py)
│   ├── region_extractor.py         # Contour detection + ROI cropping
│   ├── classifier.py               # PyTorch inference wrapper
│   ├── alert_system.py             # Alert triggers (sound, snapshot, log)
│   └── pipeline.py                 # Orchestrates the full detection pipeline
│
├── models/                         # PyTorch model definitions
│   ├── __init__.py
│   └── cnn_model.py                # CNN architecture (custom + transfer learning)
│
├── training/                       # Model training code
│   ├── train.py                    # Training loop
│   ├── dataset.py                  # Custom PyTorch Dataset class
│   └── evaluate.py                 # Evaluation metrics & confusion matrix
│
├── data/                           # Dataset storage
│   ├── raw/                        # Raw images collected from webcam
│   │   ├── person/
│   │   ├── animal/
│   │   ├── vehicle/
│   │   └── background/             # Empty/false alarm frames
│   ├── processed/                  # Resized & augmented images
│   └── splits/                     # train/val/test CSVs or folders
│
├── alerts/                         # Alert outputs
│   ├── snapshots/                  # Saved frames when intrusion detected
│   └── logs/                       # Detection logs with timestamps
│
├── utils/                          # Helper utilities
│   ├── __init__.py
│   ├── data_collector.py           # Tool to collect training data via webcam
│   └── visualizer.py               # Visualization helpers (bounding boxes, heatmaps)
│
└── notebooks/                      # Jupyter notebooks for experimentation
```

---

## 🔧 Tech Stack

| Component | Technology | Why |
|---|---|---|
| Video Capture & Processing | OpenCV (`cv2`) | You already know it, perfect for frame ops |
| Motion Detection | Frame Differencing | Your existing `object-tracking.py` code |
| Image Classification | PyTorch + torchvision | Your recently learned skill from CampusX |
| Pretrained Model | MobileNetV2 / ResNet18 | Lightweight, fast, great for transfer learning |
| Data Augmentation | torchvision.transforms | Standard pipeline, covered in CampusX |
| Alerts | `playsound`, `smtplib`, file I/O | Simple, no extra infra needed |
| Logging | Python `logging` + CSV | Track all detections with timestamps |

---

## 🗓️ Phase-wise Implementation Plan

---

### Phase 1: Project Setup & Motion Detection (Day 1-2)

**Goal**: Get the foundation running — detect motion and display bounding boxes around moving objects.

#### Step 1.1 — Environment Setup

- Create a virtual environment inside `SentinelAI/`
- Install dependencies from `requirements.txt`
- Verify webcam works with a basic OpenCV script

**Files to work on**: `requirements.txt`, `config.py`

#### Step 1.2 — Port Motion Detection Logic

Take your existing `object-tracking.py` and refactor it into a clean module.

**What to implement in `src/motion_detector.py`**:
- `MotionDetector` class with:
  - `__init__(self, scaling_factor=0.9)` — initialize the 3-frame buffer
  - `feed_frame(self, cap)` — read and preprocess a frame (grayscale, resize)
  - `detect(self)` — perform 3-frame differencing + thresholding
  - Returns: binary mask showing where motion occurred

**Key concept**: This is basically your existing code wrapped in a class so it's reusable.

```
Your existing flow:
  prev_frame, curr_frame, next_frame → absdiff → bitwise_and → threshold

Refactored flow:
  MotionDetector.feed_frame() → MotionDetector.detect() → returns binary mask
```

#### Step 1.3 — Extract Moving Regions

**What to implement in `src/region_extractor.py`**:
- `RegionExtractor` class with:
  - `extract_regions(self, binary_mask, original_frame)` — find contours in the binary mask
  - Filter out tiny contours (noise) using `cv2.contourArea() > min_area`
  - For each valid contour: get bounding box using `cv2.boundingRect()`
  - Crop the region from the **original color frame** (not the grayscale one)
  - Returns: list of cropped images + list of bounding box coordinates

**Key functions you'll use**:
```
cv2.findContours()       → find blobs in binary mask
cv2.contourArea()        → filter noise
cv2.boundingRect()       → get (x, y, w, h) of each blob
frame[y:y+h, x:x+w]     → crop the region
```

#### Step 1.4 — Test Phase 1

- Run the motion detector with webcam
- Draw bounding boxes on the live feed using `cv2.rectangle()`
- Verify that moving objects get properly bounded

**Expected output**: Live webcam feed with green rectangles around moving objects.

---

### Phase 2: Data Collection (Day 3-4)

**Goal**: Collect and organize a small dataset of images for training the classifier.

#### Step 2.1 — Build Data Collector Tool

**What to implement in `utils/data_collector.py`**:
- Script that uses your motion detector to automatically capture cropped regions
- User presses a key to assign labels:
  - `p` → save to `data/raw/person/`
  - `a` → save to `data/raw/animal/`
  - `v` → save to `data/raw/vehicle/`
  - `b` → save to `data/raw/background/`
  - `s` → skip (don't save)
- Auto-names files with timestamps: `person_20261002_185600_001.jpg`

#### Step 2.2 — Collect Data

**Target**: ~200-300 images per class (more is better, but this is enough to start)

- **Person**: Walk in front of camera from different angles, distances, clothing
- **Animal**: If you have a pet, great. Otherwise, use online datasets (see below)
- **Vehicle**: Point camera at a window/road, or use online data
- **Background**: Empty frames, curtains moving, shadows, lights changing

**Supplementary online datasets** (to boost your data):
- **CIFAR-10** (has cars, animals) — via `torchvision.datasets`
- **Caltech-101** — has person and vehicle categories
- **Google Images** — manual download for specific classes
- **Kaggle** — search "person animal vehicle classification"

#### Step 2.3 — Data Preprocessing

**What to implement in `training/dataset.py`**:
- Resize all images to a fixed size: **224×224** (standard for pretrained models)
- Create a custom `PyTorch Dataset` class:

```
class SentinelDataset(Dataset):
    __init__     → load image paths + labels
    __len__      → return total count
    __getitem__  → load image, apply transforms, return (image_tensor, label)
```

- Define transforms:
  - **Training**: RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize
  - **Validation/Test**: Only Resize + Normalize

- Split data into **train (70%) / val (15%) / test (15%)**

**Key PyTorch concepts used** (all covered in CampusX):
- `torch.utils.data.Dataset`
- `torch.utils.data.DataLoader`
- `torchvision.transforms`

---

### Phase 3: Model Building & Training (Day 5-7)

**Goal**: Train a CNN to classify cropped regions into 4 classes.

#### Step 3.1 — Define Model Architecture

**What to implement in `models/cnn_model.py`**:

**Option A — Custom CNN** (for learning):
```
Input (3×224×224)
  → Conv2d(3, 32, 3) → BatchNorm → ReLU → MaxPool
  → Conv2d(32, 64, 3) → BatchNorm → ReLU → MaxPool
  → Conv2d(64, 128, 3) → BatchNorm → ReLU → MaxPool
  → AdaptiveAvgPool2d(1)
  → Flatten
  → Linear(128, 64) → ReLU → Dropout(0.5)
  → Linear(64, 4)    # 4 classes
```

**Option B — Transfer Learning with MobileNetV2** (recommended for accuracy):
```
MobileNetV2 (pretrained on ImageNet)
  → Freeze all layers except the last block
  → Replace classifier head:
      Linear(1280, 256) → ReLU → Dropout(0.3) → Linear(256, 4)
```

> **Recommendation**: Start with Option B. MobileNetV2 is fast (good for real-time) and accurate even with small datasets. You can build Option A alongside to compare.

#### Step 3.2 — Training Loop

**What to implement in `training/train.py`**:

```
Standard PyTorch training loop:
  1. Load data using DataLoader
  2. Initialize model, loss function (CrossEntropyLoss), optimizer (Adam)
  3. For each epoch:
     a. Training pass: forward → loss → backward → step
     b. Validation pass: forward → compute accuracy
     c. Save best model checkpoint (based on val accuracy)
  4. Plot training curves (loss & accuracy vs epochs)
```

**Hyperparameters to start with**:
| Parameter | Value |
|---|---|
| Learning Rate | 0.001 (custom CNN), 0.0001 (transfer learning) |
| Batch Size | 32 |
| Epochs | 20-30 |
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |

#### Step 3.3 — Evaluate

**What to implement in `training/evaluate.py`**:
- Load best saved model
- Run on test set
- Generate:
  - **Accuracy** (overall + per-class)
  - **Confusion Matrix** (using sklearn + matplotlib)
  - **Classification Report** (precision, recall, F1 per class)
  - **Sample predictions** — show images with predicted vs actual labels

**Target accuracy**: 85%+ with transfer learning on a decent dataset.

---

### Phase 4: Integration — The Full Pipeline (Day 8-9)

**Goal**: Connect everything into a real-time detection + classification system.

#### Step 4.1 — Inference Wrapper

**What to implement in `src/classifier.py`**:
- `IntrusionClassifier` class:
  - `__init__(self, model_path)` — load trained model, set to eval mode
  - `preprocess(self, cropped_image)` — apply same transforms as training (resize, normalize)
  - `predict(self, cropped_image)` — returns predicted class + confidence score
  - Use `torch.no_grad()` during inference for speed

**Class mapping**:
```
0 → "person"     🚨 HIGH ALERT
1 → "animal"     ⚠️  MEDIUM ALERT
2 → "vehicle"    ⚠️  MEDIUM ALERT
3 → "background" ✅ IGNORE (false alarm)
```

#### Step 4.2 — Alert System

**What to implement in `src/alert_system.py`**:
- `AlertSystem` class:
  - `trigger_alert(self, alert_type, frame, bbox, confidence)`
  - Based on `alert_type`:
    - **Person detected**:
      - Save snapshot to `alerts/snapshots/` with timestamp
      - Play alert sound (optional, using `playsound`)
      - Log to `alerts/logs/detections.csv`
      - (Optional) Send email with the snapshot attached
    - **Animal/Vehicle**: Log only, no sound
    - **Background**: Ignore completely
  - **Cooldown mechanism**: Don't spam alerts — wait 30 seconds between alerts

**Log format** (`detections.csv`):
```
timestamp, class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, snapshot_path
2026-10-02 18:50:00, person, 0.94, 120, 80, 200, 350, snapshots/person_20261002_185000.jpg
```

#### Step 4.3 — Main Pipeline

**What to implement in `src/pipeline.py`** and `main.py`**:

```
Full pipeline flow:

1. Initialize webcam (cv2.VideoCapture)
2. Initialize MotionDetector
3. Initialize RegionExtractor
4. Initialize IntrusionClassifier (load trained model)
5. Initialize AlertSystem

6. LOOP:
   a. Read frame from webcam
   b. Feed to MotionDetector → get binary mask
   c. If motion detected:
      d. RegionExtractor → get cropped regions + bounding boxes
      e. For each region:
         f. IntrusionClassifier.predict() → class + confidence
         g. If confidence > threshold (e.g., 0.7):
            h. Draw bounding box on frame (color-coded by class)
            i. Put label text: "Person 94%"
            j. AlertSystem.trigger_alert()
   k. Display annotated frame
   l. Check for quit key

7. Cleanup: release camera, destroy windows
```

**Color coding for bounding boxes**:
- 🔴 Red → Person (threat)
- 🟡 Yellow → Animal
- 🔵 Blue → Vehicle
- 🟢 Green → Background/False alarm

---

### Phase 5: Polish & Extras (Day 10+)

#### 5.1 — On-Screen Dashboard (OSD)

Overlay useful info on the video feed:
- Current time
- Detection count (today)
- FPS counter
- Status: "MONITORING" / "⚠️ ALERT"
- Last detection: "Person detected at 18:50:00"

#### 5.2 — Recording Mode

- When motion is detected, start recording a video clip
- Save 5 seconds before + 10 seconds after the detection
- Store in `alerts/recordings/`

#### 5.3 — Configuration via `config.py`

Centralize all tunable parameters:
```
SCALING_FACTOR = 0.9
MIN_CONTOUR_AREA = 500
CONFIDENCE_THRESHOLD = 0.7
ALERT_COOLDOWN_SECONDS = 30
MODEL_PATH = "models/best_model.pth"
CLASSES = ["person", "animal", "vehicle", "background"]
ALERT_SOUND_PATH = "assets/alert.wav"
```

#### 5.4 — Optional Enhancements

- **Email alerts** using `smtplib` (send snapshot as attachment)
- **Telegram bot alerts** using `python-telegram-bot`
- **Web dashboard** using Flask — view live feed + detection history in browser
- **Night mode** — adjust thresholds for low-light conditions
- **Multi-camera support** — run pipeline on multiple video sources

---

## 📦 Dependencies (`requirements.txt`)

```
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
Pillow>=10.0.0
playsound>=1.3.0
```

---

## 🧠 Key PyTorch Concepts You'll Apply

Everything below was covered in the CampusX playlist:

| Concept | Where You'll Use It |
|---|---|
| `torch.nn.Module` | Defining CNN architecture |
| `Dataset` & `DataLoader` | Loading your collected images |
| `torchvision.transforms` | Data augmentation & preprocessing |
| `torchvision.models` | Loading pretrained MobileNetV2/ResNet |
| Transfer Learning | Freezing layers, replacing classifier head |
| Training Loop | Forward pass, loss, backward, optimizer step |
| `model.eval()` + `torch.no_grad()` | Fast inference in the pipeline |
| `torch.save()` / `torch.load()` | Saving & loading trained models |
| CrossEntropyLoss | Multi-class classification loss |
| Adam Optimizer | Training optimization |
| Learning Rate Scheduler | ReduceLROnPlateau for better convergence |

---

## 📊 Success Metrics

| Metric | Target |
|---|---|
| Classification Accuracy | >85% on test set |
| False Positive Rate | <10% (background classified as threat) |
| Real-time FPS | >15 FPS (usable for live monitoring) |
| Alert Response Time | <2 seconds from detection to alert |

---

## 🚀 How to Run (After Implementation)

```bash
# 1. Setup
cd SentinelAI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Collect data
python utils/data_collector.py

# 3. Train model
python training/train.py

# 4. Evaluate model
python training/evaluate.py

# 5. Run the full system
python main.py
```

---

## 📌 Tips for Building This

1. **Build incrementally** — get each phase working before moving to the next
2. **Test with videos first** — use `cv2.VideoCapture("test_video.mp4")` instead of webcam for consistent testing
3. **Start with transfer learning** — don't waste time training a custom CNN from scratch initially
4. **Keep the confidence threshold tunable** — you'll need to experiment with this
5. **Print intermediate outputs** — during development, print/display what each module produces
6. **Version control** — `git init` from day 1, commit after each phase

---

> **Next Step**: Say the word, and I'll start implementing Phase 1 — the motion detection module and region extractor!

# 🧠 NeuralForge — Visual CNN Trainer

A full end-to-end machine learning web app built with **Streamlit**.  
Train a CNN classifier in your browser with real-time metrics, confusion matrices, and live predictions.

---

## 🚀 Quick Start (VSCode)

### 1. Open Project
```
File → Open Workspace from File → neuralforge.code-workspace
```

### 2. Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Or press **F5** in VSCode (uses the launch config in the workspace file).

---

## 📁 Project Structure

```
neuralforge/
├── app.py                  ← Main entry point (sidebar + routing)
├── requirements.txt
├── neuralforge.code-workspace
│
├── pages/
│   ├── dataset.py          ← Step 1: Create classes & upload images
│   ├── train.py            ← Step 2: Configure & train CNN live
│   ├── evaluate.py         ← Step 3: Confusion matrix & metrics
│   └── predict.py          ← Step 4: Real-time prediction
│
└── utils/
    ├── session.py          ← Streamlit session state management
    ├── model_builder.py    ← Keras CNN architecture
    └── preprocessing.py    ← Image resize, normalize, augment, split
```

---

## 🔬 Tech Stack

| Library | Usage |
|---------|-------|
| **Streamlit** | Web UI framework |
| **TensorFlow / Keras** | CNN model building & training |
| **NumPy** | Array operations & preprocessing |
| **Matplotlib** | Training curves, bar charts |
| **Seaborn** | Heatmap confusion matrix |
| **scikit-learn** | train_test_split, classification_report |
| **Pillow** | Image loading & augmentation |
| **Pandas** | Results tables |

---

## 🏗️ CNN Architecture

```
Input: IMG_SIZE × IMG_SIZE × 3
────────────────────────────────
Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool
Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool
────────────────────────────────
Flatten
Dense(256) → ReLU → Dropout(0.5)
Dense(N_CLASSES) → Softmax
────────────────────────────────
Optimizer: Adam
Loss: Categorical Cross-Entropy
```

---

## 📊 Features

- **Dataset Builder** — Create classes, upload images, view thumbnails & counts
- **Data Augmentation** — Auto flip, rotate, crop, brightness jitter
- **Live Training** — Epoch-by-epoch accuracy/loss curves update in real time
- **Confusion Matrix** — Seaborn heatmap (raw + normalized)
- **Classification Report** — Precision, Recall, F1 per class
- **Overfitting Detector** — Warns if train/val gap > 20%
- **Single Predict** — Upload or use webcam, see confidence bars
- **Batch Predict** — Upload many images, get summary table + pie chart

---

## 💡 Tips

- Use **30–50 images per class** for decent accuracy
- Enable **Data Augmentation** to multiply small datasets 6×
- Use **96×96 or 128×128** for better accuracy (slower training)
- Use **64×64** for fast experimentation

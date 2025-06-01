# 🖼️ Image Caption Generator

Generate natural language descriptions for images using deep learning. This project features a modern dark mode web interface and combines a pre-trained CNN (Xception) for feature extraction with an LSTM-based language model.

## ✨ Features

- **🌙 Modern Dark Mode UI** - Beautiful, responsive web interface
- **🧠 Multiple Models** - Choose between Simple (500 images) and 8K (6000 images) models
- **📱 Drag & Drop Upload** - Easy image upload with preview
- **⚡ Real-time Generation** - Fast caption generation with loading states
- **📋 Copy & Share** - Easy caption copying and sharing
- **🎯 Model Selection** - Visual model comparison and selection

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv kenv

# Activate environment (Linux/Mac)
source kenv/bin/activate

# Activate environment (Windows)
kenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train a Model (Required)

```bash
# Quick training (10 minutes)
python train_simple.py

# Or full dataset training (2-4 hours, better results)
python train_full_8k.py
```

### 3. Start Web Interface

```bash
# Start the web server
python app.py
```

Then open your browser and go to: **http://localhost:5000**

## 🛠️ System Requirements

- **Python**: 3.7+ (3.8+ recommended)
- **RAM**: 4+ GB (8+ GB for 8K training)
- **Storage**: 2+ GB free space
- **OS**: Windows, macOS, or Linux

## 🎯 Usage

### Web Interface (Recommended)

1. **Select Model** - Choose between Simple or 8K model
2. **Upload Image** - Drag & drop or browse for an image file
3. **Generate Caption** - Click the generate button and wait for results
4. **Copy/Share** - Use the action buttons to copy or share the caption

### Command Line Interface

```bash
# Activate environment
source kenv/bin/activate

# Test with any image
python demo_simple.py path/to/your/image.jpg

# Or run interactively
python demo_simple.py
```

## 🧪 Model Comparison

| Model | Training Images | Vocabulary | Training Time | Quality |
|-------|----------------|------------|---------------|---------|
| **Simple** | 500 | 2,272 words | ~10 minutes | Good |
| **8K** | 6,000 | 8,000+ words | 2-4 hours | Excellent |

### Example Results

| Model | Example Caption |
|-------|----------------|
| **Simple** | "man in red shirt is sitting on the steps" |
| **8K** | "a man in a red shirt and jeans is sitting on concrete steps outside a building" |

## 📁 Project Structure

```
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── app.py                   # 🌟 Web application (Flask)
├── templates/               # 🌟 HTML templates
│   └── index.html          # Main web interface
├── static/                  # 🌟 Static assets
│   ├── css/style.css       # Dark mode styling
│   ├── js/main.js          # Frontend JavaScript
│   └── uploads/            # Temporary image uploads
├── train_simple.py          # Train simple model (500 images)
├── train_full_8k.py         # Train 8K model (6000 images)
├── demo_simple.py           # Command line interface
├── models/                  # Trained models
│   ├── model_simple.h5      # Simple model weights
│   ├── tokenizer_simple.pkl # Simple model tokenizer
│   ├── model_8k_best.h5     # 8K model weights (after training)
│   └── tokenizer_8k.pkl     # 8K model tokenizer (after training)
├── Flickr8k_Dataset/        # 8,091 training images
├── descriptions.txt         # Image captions dataset
└── features.p              # Pre-extracted image features
```

## 🔧 Training Models

### Simple Model (Quick Start)
```bash
python train_simple.py
# ⏱️ Time: ~10 minutes
# 📊 Data: 500 images, 5 epochs
# 💾 Output: model_simple.h5 + tokenizer_simple.pkl
```

### 8K Model (Best Quality)
```bash
python train_full_8k.py
# ⏱️ Time: 2-4 hours
# 📊 Data: 6,000 images, 20 epochs
# 💾 Output: model_8k_best.h5 + tokenizer_8k.pkl
# ⚠️ Requires: 8+ GB RAM
```

## 🛠️ Technical Details

### Model Architecture
- **Image Encoder**: Xception CNN (pre-trained on ImageNet)
- **Text Decoder**: LSTM with 256 hidden units
- **Input**: 299×299 RGB images
- **Output**: Natural language captions

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)

## 🚧 Troubleshooting

### Common Issues

**No models available:**
```bash
# Train a model first
python train_simple.py
```

**Memory errors during training:**
```bash
# Reduce batch size in training scripts
# Edit train_simple.py: batch_size = 32
```

**TensorFlow installation issues:**
```bash
# For CPU-only (recommended)
pip install tensorflow-cpu

# For GPU support (requires CUDA)
pip install tensorflow-gpu
```

## 🎓 How It Works

1. **Feature Extraction**: Xception CNN extracts 2048-dim features from images
2. **Sequence Generation**: LSTM predicts next word given image features + previous words
3. **Training**: Model learns from 40,000+ image-caption pairs
4. **Inference**: Generate captions word-by-word until "end" token

## 📊 Project Files

### Core Files
- **`app.py`** - Main web application
- **`train_simple.py`** - Train simple model (500 images)
- **`train_full_8k.py`** - Train 8K model (6000 images)
- **`templates/index.html`** - Web interface
- **`static/`** - CSS, JavaScript, and assets

## 🚧 Limitations

### Simple Model
- Generic descriptions for complex scenes
- Limited vocabulary (2,272 words)
- Best with common subjects (people, animals, outdoor scenes)

### 8K Model (After Training)
- Much more detailed and accurate descriptions
- Larger vocabulary (8,000+ words)
- Better object recognition and scene understanding

---

## 🚀 Ready to Start?

1. **Train a model**: `python train_simple.py`
2. **Start web interface**: `python app.py`
3. **Open browser**: http://localhost:5000
4. **Upload image and generate captions!**

🖼️➡️📝 **Happy captioning!**

---

*Built with TensorFlow, Flask, and ❤️*

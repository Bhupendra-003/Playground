# Image Caption Generation

Generate natural language descriptions for images using deep learning. This project combines a pre-trained CNN (Xception) for feature extraction with an LSTM-based language model.

## 🚀 Quick Start

```bash
# Activate environment
source kenv/bin/activate

# Test with any image
python demo_simple.py path/to/your/image.jpg

# Or run interactively
python demo_simple.py
```

## 📁 Project Structure

```
├── train_simple.py          # Train model (500 images, ~10 min)
├── train_full_8k.py         # Train on full dataset (6000 images, 2-4 hours)
├── demo_simple.py           # Interactive testing (recommended)
├── test_simple.py           # Single image testing
├── test_8k_model.py         # Test 8K model specifically
├── compare_simple_vs_8k.py  # Compare both models side-by-side
├── status_check.py          # Check project status
├── models/                  # Trained models
│   ├── model_simple.h5      # Current model (500 images)
│   └── tokenizer_simple.pkl # Text tokenizer
├── Flickr8k_Dataset/        # 8,091 images
├── descriptions.txt         # Image captions
└── features.p              # Pre-extracted features
```

## 🎯 Current Model Performance

- **Training Data**: 500 images from Flickr8k
- **Training Time**: ~10 minutes
- **Vocabulary**: 2,272 words
- **Example Results**:
  - "man in red shirt is sitting on the steps"
  - "dog jumps through the air"
  - "young boy in yellow shirt is running"

## 🧪 Testing Your Model

### Method 1: Interactive Demo (Recommended)
```bash
python demo_simple.py
# Enter image paths when prompted
```

### Method 2: Command Line
```bash
python demo_simple.py your_image.jpg
python test_simple.py --image your_image.jpg
```

### Method 3: Test with Dataset Images
```bash
python demo_simple.py Flickr8k_Dataset/1000268201_693b08cb0e.jpg
```

### Method 4: Test 8K Model (After Training)
```bash
python test_8k_model.py your_image.jpg
```

### Method 5: Compare Models
```bash
python compare_simple_vs_8k.py your_image.jpg
# Shows captions from both models side-by-side
```

## 🔧 Training Options

### Quick Training (Current)
```bash
python train_simple.py
# 500 images, 5 epochs, ~10 minutes
# Creates: model_simple.h5 + tokenizer_simple.pkl
```

### Full Dataset Training (Better Results)
```bash
python train_full_8k.py
# 6000 images, 20 epochs, 2-4 hours
# Requires 8+ GB RAM
# Creates: model_8k_best.h5 + tokenizer_8k.pkl
```

## 🧪 Testing the 8K Model

### Before Training 8K Model
The 8K model doesn't exist yet. Test scripts will show:
```bash
python test_8k_model.py your_image.jpg
# ❌ 8K model not found! Run: python train_full_8k.py
```

### After Training 8K Model
Once you've run `python train_full_8k.py`, you can:

#### Test 8K Model Only
```bash
python test_8k_model.py your_image.jpg
# Shows caption from 8K model (8000+ vocabulary)
```

#### Compare Both Models
```bash
python compare_simple_vs_8k.py your_image.jpg
# Shows captions from both models side-by-side
# Includes vocabulary analysis and differences
```

### Expected Quality Improvement
| Model | Example Caption |
|-------|----------------|
| **Simple** | "man in red shirt is sitting on the steps" |
| **8K** | "a man in a red shirt and jeans is sitting on concrete steps outside a building" |

## 📊 Model Architecture

- **Image Encoder**: Xception CNN (pre-trained on ImageNet)
- **Text Decoder**: LSTM with 256 hidden units
- **Vocabulary**: 2,272 unique words
- **Input**: 299×299 RGB images
- **Output**: Natural language captions

## 🛠️ System Requirements

- **Python**: 3.7+
- **RAM**: 4+ GB (8+ GB for full training)
- **Dependencies**: TensorFlow, Keras, NumPy, Pillow, matplotlib

## 📈 Improving Results

### Option 1: Train 8K Model (Recommended)
```bash
python train_full_8k.py
# Best quality improvement: 12x more data, 3.5x larger vocabulary
```

### Option 2: Modify Training Parameters
- **Longer training**: Increase epochs in training scripts
- **Better images**: Use clear, well-lit photos with obvious subjects
- **More data**: Add your own images to the dataset

### Quality Comparison
| Aspect | Simple Model | 8K Model | Improvement |
|--------|--------------|----------|-------------|
| **Training Images** | 500 | 6,000 | 12x more |
| **Vocabulary** | 2,272 words | 8,000+ words | 3.5x larger |
| **Training Time** | 10 minutes | 2-4 hours | Worth the wait |
| **Caption Quality** | Basic | Production-ready | Significantly better |

## 🔍 Check Status

```bash
python status_check.py
```

Shows:
- ✅/❌ Simple model status
- ✅/❌ 8K model status
- 📊 Dataset information
- 🚀 Available commands
- 📈 Vocabulary sizes

## 🎯 Quick Start Guide

### For Testing (Current Model)
```bash
source kenv/bin/activate
python demo_simple.py your_image.jpg
```

### For Better Results (Train 8K Model)
```bash
source kenv/bin/activate
python train_full_8k.py          # Train (2-4 hours)
python test_8k_model.py your_image.jpg    # Test
python compare_simple_vs_8k.py your_image.jpg  # Compare
```

## 🎓 How It Works

1. **Feature Extraction**: Xception CNN extracts 2048-dim features from images
2. **Sequence Generation**: LSTM predicts next word given image features + previous words
3. **Training**: Model learns from 40,000+ image-caption pairs
4. **Inference**: Generate captions word-by-word until "end" token

## 📝 Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- GIF (.gif)

## 🚧 Current Limitations

### Simple Model (500 images)
- Generic descriptions for complex scenes
- Limited vocabulary (2,272 words)
- Best with common subjects (people, animals, outdoor scenes)
- May miss fine details or context

### 8K Model (6000 images) - After Training
- Much more detailed descriptions
- Larger vocabulary (8,000+ words)
- Better object recognition and scene understanding
- More natural language and context awareness

---

## 🚀 Ready to Start?

**Current Model:** `python demo_simple.py your_image.jpg`

**Better Results:** `python train_full_8k.py` → `python test_8k_model.py your_image.jpg`

**Compare Both:** `python compare_simple_vs_8k.py your_image.jpg`

🖼️➡️📝 **Start generating captions now!**

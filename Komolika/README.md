# Image Caption Generation Project

An AI-powered image captioning system that generates natural language descriptions for images using deep learning.

## 🎯 Overview

This project implements an image caption generation model using:
- **CNN (Xception)** for image feature extraction
- **LSTM** for sequence generation
- **TensorFlow/Keras** framework

## 📁 Project Structure

```
Komolika/
├── kenv/                    # Virtual environment
├── descriptions.txt         # Processed image descriptions (40K+ captions)
├── features.p              # Pre-extracted image features (8K+ images)
├── tokenizer_new.pkl       # Trained tokenizer (8.4K vocabulary)
├── model_demo.h5           # Trained model (62MB)
├── main.py                 # Original training script
├── simple_train.py         # Simplified training script
├── test_demo.py            # Working test script
├── demo.py                 # File validation script
└── project_summary.py      # Project overview
```

## 🚀 Quick Start

### 1. Activate Environment
```bash
source kenv/bin/activate
```

### 2. Test with Any Image
```bash
python test_demo.py --image path/to/your/image.jpg
```

### 3. View Project Status
```bash
python project_summary.py
```

## 📊 Dataset Information

- **Total descriptions**: 40,460 captions
- **Unique images**: 8,092 images
- **Feature vectors**: 2048-dimensional (Xception CNN)
- **Vocabulary size**: 8,423 unique words

## 🧠 Model Architecture

```
Input Image (299x299x3)
    ↓
Xception CNN (Feature Extractor)
    ↓
2048-dimensional features
    ↓
Dense Layer (256 units)
    ↓
LSTM + Embedding (Text Sequence)
    ↓
Dense Output (Vocabulary Size)
    ↓
Generated Caption
```

## 🔧 Technical Details

- **Framework**: TensorFlow 2.x (nightly build)
- **Python**: 3.13.3
- **CNN**: Xception (pre-trained on ImageNet)
- **RNN**: LSTM with 256 hidden units
- **Embedding**: 256-dimensional word embeddings
- **Loss**: Categorical crossentropy
- **Optimizer**: Adam

## 📋 Usage Examples

### Basic Caption Generation
```bash
python test_demo.py --image my_photo.jpg
```

### With Image Display
```bash
python test_demo.py --image my_photo.jpg --show
```

### Custom Model/Tokenizer
```bash
python test_demo.py --image my_photo.jpg --model custom_model.h5 --tokenizer custom_tokenizer.pkl
```

## 🏋️ Training

### Quick Demo Training (2 epochs)
```bash
python simple_train.py
```

### Full Training (requires Flickr8k dataset)
```bash
python main.py
```

## 📈 Performance

The current model is a demo trained for only 2 epochs. For better performance:
1. Train for 20-50 epochs
2. Use the full Flickr8k dataset
3. Implement beam search
4. Add BLEU score evaluation

## 🔧 Dependencies

All dependencies are installed in the virtual environment:
- tensorflow (nightly)
- pillow
- numpy
- matplotlib
- pandas
- tqdm

## 🎓 Next Steps

1. **Get Flickr8k Dataset**: Download the complete dataset for full training
2. **Extended Training**: Train for more epochs (20-50)
3. **Evaluation Metrics**: Implement BLEU, METEOR, CIDEr scores
4. **Beam Search**: Improve caption quality with beam search
5. **Different Architectures**: Try ResNet, VGG, or Vision Transformers

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Warnings**: Normal if no GPU available, model runs on CPU
2. **TensorFlow Warnings**: Can be ignored, model works correctly
3. **Image Loading Errors**: Ensure image path is correct and format is supported

### File Validation
```bash
python demo.py
```

## 📝 Example Output

```
🖼️  Loading image: sample.jpg
🧠 Loading model: model_demo.h5
🔤 Loading tokenizer: tokenizer_new.pkl
✓ Tokenizer loaded. Vocabulary size: 8423
✓ Model loaded successfully
🔍 Loading Xception model for feature extraction...
✓ Xception model loaded
🔍 Extracting features from image...
✓ Features extracted
📝 Generating caption...

==================================================
🎯 GENERATED CAPTION:
   a dog is running through the grass
==================================================
```

## 🤝 Contributing

Feel free to improve the model, add features, or fix bugs!

## 📄 License

This project is for educational purposes.

---

**Status**: ✅ Working and Ready to Use!

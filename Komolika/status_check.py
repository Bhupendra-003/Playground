#!/usr/bin/env python3
"""
Status check script for the image caption generation project
"""

import os
import sys
from pickle import load

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print(f"✅ {description}: {filepath} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_contents(dirpath, description):
    """Check directory contents"""
    if os.path.exists(dirpath):
        files = os.listdir(dirpath)
        print(f"✅ {description}: {dirpath} ({len(files)} files)")
        return True
    else:
        print(f"❌ {description}: {dirpath} (NOT FOUND)")
        return False

def main():
    print("🔍 IMAGE CAPTION GENERATION PROJECT STATUS CHECK")
    print("=" * 60)
    
    # Check trained model files
    print("\n📁 TRAINED MODEL FILES:")
    model_exists = check_file_exists("models/model_simple.h5", "Trained Model")
    tokenizer_exists = check_file_exists("models/tokenizer_simple.pkl", "Tokenizer")
    
    # Check dataset files
    print("\n📁 DATASET FILES:")
    check_file_exists("descriptions.txt", "Image Descriptions")
    check_file_exists("features.p", "Pre-extracted Features")
    check_directory_contents("Flickr8k_Dataset", "Image Dataset")
    check_directory_contents("Flickr8k_text", "Text Files")
    
    # Check script files
    print("\n📁 SCRIPT FILES:")
    check_file_exists("train_simple.py", "Training Script")
    check_file_exists("test_simple.py", "Test Script")
    check_file_exists("demo_simple.py", "Demo Script")
    
    # Check if model is ready to use
    print("\n🎯 MODEL STATUS:")
    if model_exists and tokenizer_exists:
        print("✅ Model is ready for inference!")
        
        # Try to load tokenizer to get vocab info
        try:
            tokenizer = load(open("models/tokenizer_simple.pkl", "rb"))
            vocab_size = len(tokenizer.word_index) + 1
            print(f"✅ Vocabulary size: {vocab_size:,} words")
        except Exception as e:
            print(f"⚠️  Could not load tokenizer details: {e}")
        
        print("\n🚀 READY TO USE:")
        print("   python test_simple.py --image path/to/your/image.jpg")
        print("   python demo_simple.py")
        
    else:
        print("❌ Model not ready. Please run training first:")
        print("   python train_simple.py")
    
    # Check environment
    print("\n🔧 ENVIRONMENT:")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not available")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not available")
    
    try:
        from PIL import Image
        print("✅ PIL (Pillow) available")
    except ImportError:
        print("❌ PIL (Pillow) not available")
    
    # Summary
    print("\n" + "=" * 60)
    if model_exists and tokenizer_exists:
        print("🎉 PROJECT STATUS: READY FOR INFERENCE!")
        print("📝 See TRAINING_SUMMARY.md for detailed results")
    else:
        print("⚠️  PROJECT STATUS: NEEDS TRAINING")
        print("📝 Run 'python train_simple.py' to train the model")
    
    print("=" * 60)

if __name__ == '__main__':
    main()

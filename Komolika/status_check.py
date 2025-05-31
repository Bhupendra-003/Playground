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
    model_exists = check_file_exists("models/model_simple.h5", "Simple Model (500 images)")
    tokenizer_exists = check_file_exists("models/tokenizer_simple.pkl", "Simple Tokenizer")

    # Check 8k model files
    model_8k_exists = check_file_exists("models/model_8k_best.h5", "8K Model (6000 images)")
    tokenizer_8k_exists = check_file_exists("models/tokenizer_8k.pkl", "8K Tokenizer")
    
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
    
    # Check if models are ready to use
    print("\n🎯 MODEL STATUS:")

    if model_exists and tokenizer_exists:
        print("✅ Simple Model is ready for inference!")

        # Try to load simple tokenizer to get vocab info
        try:
            tokenizer = load(open("models/tokenizer_simple.pkl", "rb"))
            vocab_size = len(tokenizer.word_index) + 1
            print(f"✅ Simple Model vocabulary: {vocab_size:,} words")
        except Exception as e:
            print(f"⚠️  Could not load simple tokenizer details: {e}")
    else:
        print("❌ Simple Model not ready. Run: python train_simple.py")

    if model_8k_exists and tokenizer_8k_exists:
        print("✅ 8K Model is ready for inference!")

        # Try to load 8k tokenizer to get vocab info
        try:
            tokenizer_8k = load(open("models/tokenizer_8k.pkl", "rb"))
            vocab_size_8k = len(tokenizer_8k.word_index) + 1
            print(f"✅ 8K Model vocabulary: {vocab_size_8k:,} words")
        except Exception as e:
            print(f"⚠️  Could not load 8K tokenizer details: {e}")
    else:
        print("❌ 8K Model not ready. Run: python train_full_8k.py")

    print("\n🚀 AVAILABLE COMMANDS:")
    if model_exists and tokenizer_exists:
        print("   python test_simple.py --image path/to/your/image.jpg")
        print("   python demo_simple.py")

    if model_8k_exists and tokenizer_8k_exists:
        print("   python test_8k.py --image path/to/your/image.jpg")

    if model_exists and model_8k_exists:
        print("   python compare_models.py --image path/to/your/image.jpg")

    if not (model_exists or model_8k_exists):
        print("   python train_simple.py (quick training)")
        print("   python train_full_8k.py (full dataset training)")
    
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
        print("🎉 PROJECT STATUS: SIMPLE MODEL READY!")
        if model_8k_exists and tokenizer_8k_exists:
            print("🎉 BONUS: 8K MODEL ALSO READY!")
            print("📝 Use compare_models.py to see the difference")
        else:
            print("📝 Run 'python train_full_8k.py' for better results")
    elif model_8k_exists and tokenizer_8k_exists:
        print("🎉 PROJECT STATUS: 8K MODEL READY!")
        print("📝 High-quality model trained on full dataset")
    else:
        print("⚠️  PROJECT STATUS: NEEDS TRAINING")
        print("📝 Run 'python train_simple.py' for quick start")
        print("📝 Or 'python train_full_8k.py' for best results")

    print("=" * 60)

if __name__ == '__main__':
    main()

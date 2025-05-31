#!/usr/bin/env python3
"""
Simple demo script for image caption generation
"""

import os
import sys
import argparse
import numpy as np
from pickle import load
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

def extract_features(filename, model):
    """Extract features from an image using Xception model"""
    try:
        image = Image.open(filename)
    except Exception as e:
        print(f"ERROR: Couldn't open image '{filename}': {e}")
        return None
    
    # Resize to 299x299 for Xception
    image = image.resize((299, 299))
    image = np.array(image)
    
    # Handle images with 4 channels (RGBA)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[..., :3]
    
    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Expand dimensions and preprocess
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image = preprocess_input(image)
    
    # Extract features
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    """Get word for given integer ID"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    """Generate description for an image"""
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def define_model(vocab_size, max_length):
    """Define the image captioning model architecture"""
    # Feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Tie it together
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    model_path = 'models/model_simple.h5'
    tokenizer_path = 'models/tokenizer_simple.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run 'python train_simple.py' first to train the model.")
        return None, None, None, None
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        print("Please run 'python train_simple.py' first to train the model.")
        return None, None, None, None
    
    # Load tokenizer
    try:
        tokenizer = load(open(tokenizer_path, "rb"))
        vocab_size = len(tokenizer.word_index) + 1
        max_length = 35
        print(f"✓ Tokenizer loaded. Vocabulary size: {vocab_size:,}")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None, None, None, None
    
    # Create model architecture and load weights
    try:
        print("🏗️  Creating model architecture...")
        model = define_model(vocab_size, max_length)
        print("✓ Model architecture created")
        
        print("📥 Loading model weights...")
        model.load_weights(model_path)
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None
    
    # Load Xception model for feature extraction
    print("🔍 Loading Xception model for feature extraction...")
    xception_model = Xception(include_top=False, pooling="avg")
    print("✓ Xception model loaded")
    
    return model, tokenizer, xception_model, max_length

def generate_caption_for_image(image_path, model, tokenizer, xception_model, max_length):
    """Generate caption for a single image"""
    print(f"\n📸 Processing: {image_path}")
    
    # Extract features from the image
    print("🔍 Extracting features...")
    photo = extract_features(image_path, xception_model)
    if photo is None:
        return None
    
    # Generate caption
    print("📝 Generating caption...")
    description = generate_desc(model, tokenizer, photo, max_length)
    
    # Clean up the description
    description = description.replace('start ', '').replace(' end', '')
    
    return description

def main():
    print("🖼️  IMAGE CAPTION GENERATION DEMO")
    print("=" * 50)
    
    # Load model and tokenizer
    model, tokenizer, xception_model, max_length = load_model_and_tokenizer()
    if model is None:
        return
    
    print("\n🎯 Model loaded successfully!")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Command line mode
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        caption = generate_caption_for_image(image_path, model, tokenizer, xception_model, max_length)
        if caption:
            print(f"\n🎯 CAPTION: {caption}")
    else:
        # Interactive mode
        print("\n💡 Usage:")
        print("  python demo_simple.py <image_path>")
        print("  or run interactively:")
        print()
        
        while True:
            try:
                image_path = input("📁 Enter image path (or 'quit' to exit): ").strip()
                
                if image_path.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not image_path:
                    continue
                
                if not os.path.exists(image_path):
                    print(f"❌ Image not found: {image_path}")
                    continue
                
                caption = generate_caption_for_image(image_path, model, tokenizer, xception_model, max_length)
                if caption:
                    print(f"\n🎯 CAPTION: {caption}")
                    print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()

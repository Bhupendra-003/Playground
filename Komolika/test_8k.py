#!/usr/bin/env python3
"""
Test script for the 8k-trained model
"""

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
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
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

def main():
    parser = argparse.ArgumentParser(description='Generate caption for an image using 8k-trained model')
    parser.add_argument('-i', '--image', required=True, help='Path to the image file')
    parser.add_argument('--model', default='models/model_8k_best.h5', help='Path to the trained model')
    parser.add_argument('--tokenizer', default='models/tokenizer_8k.pkl', help='Path to the tokenizer')
    parser.add_argument('--max_length', type=int, default=35, help='Maximum caption length')
    parser.add_argument('--show', action='store_true', help='Display the image')
    
    args = parser.parse_args()
    
    print("🖼️  IMAGE CAPTION GENERATION (8K-Trained Model)")
    print("=" * 60)
    print(f"📁 Image: {args.image}")
    print(f"🧠 Model: {args.model}")
    print(f"🔤 Tokenizer: {args.tokenizer}")
    
    # Load tokenizer
    try:
        tokenizer = load(open(args.tokenizer, "rb"))
        vocab_size = len(tokenizer.word_index) + 1
        print(f"✓ Tokenizer loaded. Vocabulary size: {vocab_size:,}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        print("Make sure you've trained the 8k model first: python train_full_8k.py")
        return
    
    # Create model architecture and load weights
    try:
        print("🏗️  Creating model architecture...")
        model = define_model(vocab_size, args.max_length)
        print("✓ Model architecture created")
        
        print("📥 Loading model weights...")
        model.load_weights(args.model)
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure you've trained the 8k model first: python train_full_8k.py")
        return
    
    # Load Xception model for feature extraction
    print("🔍 Loading Xception model for feature extraction...")
    xception_model = Xception(include_top=False, pooling="avg")
    print("✓ Xception model loaded")
    
    # Extract features from the image
    print("🔍 Extracting features from image...")
    photo = extract_features(args.image, xception_model)
    if photo is None:
        return
    print("✓ Features extracted")
    
    # Generate caption
    print("📝 Generating caption...")
    description = generate_desc(model, tokenizer, photo, args.max_length)
    
    # Clean up the description
    description = description.replace('start ', '').replace(' end', '')
    
    print("\n" + "="*60)
    print("🎯 GENERATED CAPTION (8K Model):")
    print(f"   {description}")
    print("="*60)
    
    # Display image if requested
    if args.show:
        try:
            img = Image.open(args.image)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.title(f"8K Model Caption: {description}", fontsize=14, wrap=True)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")

if __name__ == '__main__':
    main()

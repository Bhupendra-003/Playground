#!/usr/bin/env python3
"""
Test script specifically for the 8K-trained model
"""

import os
import sys
import numpy as np
from pickle import load
from PIL import Image
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
    print("🚀 8K MODEL TESTER")
    print("=" * 50)
    
    # Check if 8K model exists
    if not os.path.exists('models/model_8k_best.h5'):
        print("❌ 8K model not found!")
        print()
        print("The 8K model hasn't been trained yet.")
        print("To train it, run:")
        print("  python train_full_8k.py")
        print()
        print("This will:")
        print("  - Train on 6,000 images (vs 500 in simple model)")
        print("  - Take 2-4 hours")
        print("  - Create a much better model with 8,000+ vocabulary")
        print("  - Produce more detailed and accurate captions")
        print()
        print("After training, you can test it with:")
        print("  python test_8k_model.py your_image.jpg")
        return
    
    if not os.path.exists('models/tokenizer_8k.pkl'):
        print("❌ 8K tokenizer not found!")
        return
    
    if len(sys.argv) < 2:
        print("Usage: python test_8k_model.py <image_path>")
        print()
        print("Examples:")
        print("  python test_8k_model.py Flickr8k_Dataset/1000268201_693b08cb0e.jpg")
        print("  python test_8k_model.py your_own_image.jpg")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        # Load tokenizer
        tokenizer = load(open('models/tokenizer_8k.pkl', "rb"))
        vocab_size = len(tokenizer.word_index) + 1
        max_length = 35
        
        print(f"✓ 8K Tokenizer loaded. Vocabulary size: {vocab_size:,}")
        
        # Create and load model
        print("🏗️  Creating 8K model architecture...")
        model = define_model(vocab_size, max_length)
        print("✓ Model architecture created")
        
        print("📥 Loading 8K model weights...")
        model.load_weights('models/model_8k_best.h5')
        print("✓ 8K model weights loaded successfully")
        
        # Load Xception model for feature extraction
        print("🔍 Loading Xception model for feature extraction...")
        xception_model = Xception(include_top=False, pooling="avg")
        print("✓ Xception model loaded")
        
        # Extract features
        print(f"📸 Processing: {image_path}")
        print("🔍 Extracting features...")
        photo = extract_features(image_path, xception_model)
        if photo is None:
            return
        
        # Generate caption
        print("📝 Generating caption...")
        description = generate_desc(model, tokenizer, photo, max_length)
        
        # Clean up the description
        description = description.replace('start ', '').replace(' end', '')
        
        print("\n" + "="*60)
        print("🎯 8K MODEL CAPTION:")
        print(f"   {description}")
        print("="*60)
        print(f"📊 Model trained on 6,000 images with {vocab_size:,} vocabulary")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()

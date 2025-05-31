#!/usr/bin/env python3
"""
Compare captions from Simple Model (500 images) vs 8K Model (6000 images)
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

def load_model_and_tokenizer(model_path, tokenizer_path, model_name):
    """Load a model and tokenizer"""
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None, None, None, None
    
    try:
        # Load tokenizer
        tokenizer = load(open(tokenizer_path, "rb"))
        vocab_size = len(tokenizer.word_index) + 1
        max_length = 35
        
        # Create and load model
        model = define_model(vocab_size, max_length)
        model.load_weights(model_path)
        
        return model, tokenizer, vocab_size, max_length
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None, None, None

def main():
    print("🔍 MODEL COMPARISON: Simple vs 8K")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python compare_simple_vs_8k.py <image_path>")
        print()
        print("Examples:")
        print("  python compare_simple_vs_8k.py Flickr8k_Dataset/1000268201_693b08cb0e.jpg")
        print("  python compare_simple_vs_8k.py your_own_image.jpg")
        print()
        print("This script compares captions from:")
        print("  🔹 Simple Model: 500 images, ~2,272 vocabulary")
        print("  🔸 8K Model: 6,000 images, ~8,000+ vocabulary")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📁 Image: {image_path}")
    print("=" * 60)
    
    # Load both models
    print("📥 Loading models...")
    
    simple_model, simple_tokenizer, simple_vocab, simple_max_len = load_model_and_tokenizer(
        'models/model_simple.h5', 'models/tokenizer_simple.pkl', 'Simple Model'
    )
    
    model_8k, tokenizer_8k, vocab_8k, max_len_8k = load_model_and_tokenizer(
        'models/model_8k_best.h5', 'models/tokenizer_8k.pkl', '8K Model'
    )
    
    # Check if we have at least one model
    if simple_model is None and model_8k is None:
        print("❌ No models found!")
        print()
        print("Train at least one model:")
        print("  python train_simple.py (quick, 10 minutes)")
        print("  python train_full_8k.py (better quality, 2-4 hours)")
        return
    
    # Load Xception model for feature extraction
    print("🔍 Loading Xception model for feature extraction...")
    xception_model = Xception(include_top=False, pooling="avg")
    
    # Extract features from the image
    print("🔍 Extracting features from image...")
    photo = extract_features(image_path, xception_model)
    if photo is None:
        return
    
    print("\n" + "="*60)
    print("🎯 CAPTION COMPARISON:")
    print("="*60)
    
    # Generate caption with simple model
    if simple_model is not None:
        print("📝 Generating caption with Simple Model...")
        simple_caption = generate_desc(simple_model, simple_tokenizer, photo, simple_max_len)
        simple_caption = simple_caption.replace('start ', '').replace(' end', '')
        print(f"🔹 Simple Model (500 images, {simple_vocab:,} words):")
        print(f"   {simple_caption}")
    else:
        print("🔹 Simple Model: Not available")
        print("   Run: python train_simple.py")
    
    print()
    
    # Generate caption with 8k model
    if model_8k is not None:
        print("📝 Generating caption with 8K Model...")
        caption_8k = generate_desc(model_8k, tokenizer_8k, photo, max_len_8k)
        caption_8k = caption_8k.replace('start ', '').replace(' end', '')
        print(f"🔸 8K Model (6000 images, {vocab_8k:,} words):")
        print(f"   {caption_8k}")
    else:
        print("🔸 8K Model: Not available")
        print("   Run: python train_full_8k.py")
    
    print("="*60)
    
    # Show analysis if both models are available
    if simple_model is not None and model_8k is not None:
        print("\n📊 ANALYSIS:")
        print(f"   Vocabulary difference: {vocab_8k - simple_vocab:,} more words in 8K model")
        
        simple_words = set(simple_caption.split())
        model_8k_words = set(caption_8k.split())
        
        print(f"   Simple model caption: {len(simple_words)} unique words")
        print(f"   8K model caption: {len(model_8k_words)} unique words")
        
        if simple_words != model_8k_words:
            print("   ✓ Models generated different captions")
            
            # Show unique words
            only_in_8k = model_8k_words - simple_words
            only_in_simple = simple_words - model_8k_words
            
            if only_in_8k:
                print(f"   🔸 Words only in 8K model: {', '.join(sorted(only_in_8k))}")
            if only_in_simple:
                print(f"   🔹 Words only in Simple model: {', '.join(sorted(only_in_simple))}")
        else:
            print("   ⚠️ Models generated identical captions")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Compare captions from simple model (500 images) vs 8k model (full dataset)
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
import os

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
    print(f"📥 Loading {model_name}...")
    
    if not os.path.exists(model_path):
        print(f"❌ {model_name} not found at {model_path}")
        return None, None, None, None
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer for {model_name} not found at {tokenizer_path}")
        return None, None, None, None
    
    try:
        # Load tokenizer
        tokenizer = load(open(tokenizer_path, "rb"))
        vocab_size = len(tokenizer.word_index) + 1
        max_length = 35
        
        # Create and load model
        model = define_model(vocab_size, max_length)
        model.load_weights(model_path)
        
        print(f"✓ {model_name} loaded (vocab: {vocab_size:,} words)")
        return model, tokenizer, vocab_size, max_length
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description='Compare captions from simple vs 8k models')
    parser.add_argument('-i', '--image', required=True, help='Path to the image file')
    parser.add_argument('--show', action='store_true', help='Display the image')
    
    args = parser.parse_args()
    
    print("🔍 MODEL COMPARISON: Simple (500 images) vs 8K (full dataset)")
    print("=" * 70)
    print(f"📁 Image: {args.image}")
    
    # Load both models
    simple_model, simple_tokenizer, simple_vocab, simple_max_len = load_model_and_tokenizer(
        'models/model_simple.h5', 'models/tokenizer_simple.pkl', 'Simple Model (500 images)'
    )
    
    model_8k, tokenizer_8k, vocab_8k, max_len_8k = load_model_and_tokenizer(
        'models/model_8k_best.h5', 'models/tokenizer_8k.pkl', '8K Model (full dataset)'
    )
    
    # Check if we have at least one model
    if simple_model is None and model_8k is None:
        print("❌ No models found! Please train at least one model first.")
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
    
    print("\n" + "="*70)
    print("🎯 CAPTION COMPARISON:")
    print("="*70)
    
    # Generate caption with simple model
    if simple_model is not None:
        print("📝 Generating caption with Simple Model...")
        simple_caption = generate_desc(simple_model, simple_tokenizer, photo, simple_max_len)
        simple_caption = simple_caption.replace('start ', '').replace(' end', '')
        print(f"🔹 Simple Model (500 images, 5 epochs):")
        print(f"   {simple_caption}")
        print(f"   Vocabulary: {simple_vocab:,} words")
    else:
        print("🔹 Simple Model: Not available")
        print("   Run: python train_simple.py")
    
    print()
    
    # Generate caption with 8k model
    if model_8k is not None:
        print("📝 Generating caption with 8K Model...")
        caption_8k = generate_desc(model_8k, tokenizer_8k, photo, max_len_8k)
        caption_8k = caption_8k.replace('start ', '').replace(' end', '')
        print(f"🔸 8K Model (6000 images, 20 epochs):")
        print(f"   {caption_8k}")
        print(f"   Vocabulary: {vocab_8k:,} words")
    else:
        print("🔸 8K Model: Not available")
        print("   Run: python train_full_8k.py")
    
    print("="*70)
    
    # Show analysis if both models are available
    if simple_model is not None and model_8k is not None:
        print("\n📊 ANALYSIS:")
        print(f"   Vocabulary difference: {vocab_8k - simple_vocab:,} more words in 8K model")
        
        simple_words = set(simple_caption.split())
        model_8k_words = set(caption_8k.split())
        
        print(f"   Simple model caption length: {len(simple_words)} unique words")
        print(f"   8K model caption length: {len(model_8k_words)} unique words")
        
        if simple_words != model_8k_words:
            print("   ✓ Models generated different captions")
        else:
            print("   ⚠️ Models generated identical captions")
    
    # Display image if requested
    if args.show:
        try:
            img = Image.open(args.image)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            
            title_parts = []
            if simple_model is not None:
                title_parts.append(f"Simple: {simple_caption}")
            if model_8k is not None:
                title_parts.append(f"8K: {caption_8k}")
            
            plt.title("\n".join(title_parts), fontsize=12, wrap=True)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")

if __name__ == '__main__':
    main()

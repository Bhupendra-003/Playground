#!/usr/bin/env python3
"""
Simple training script for 1000 images using batch training instead of generators
"""

import os
import string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pickle import dump, load
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

def all_img_captions(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    captions = {}
    for line in text.split('\n'):
        tokens = line.split('\t')
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1]
        image_id = image_id.split('.')[0]
        if image_id not in captions:
            captions[image_id] = []
        captions[image_id].append(image_desc)
    return captions

def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img_id, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption = img_caption.replace('-', ' ')
            desc = img_caption.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            img_caption = ' '.join(desc)
            captions[img_id][i] = img_caption
    return captions

def load_photos(filename, dataset_images):
    file = open(filename, 'r')
    photos = file.read().split('\n')[:-1]
    file.close()
    return photos

def load_clean_descriptions(filename, photos):
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    descriptions = {}
    for line in doc.split('\n'):
        tokens = line.split('\t')
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1]
        if image_id in photos:
            if image_id not in descriptions:
                descriptions[image_id] = []
            descriptions[image_id].append('start ' + image_desc + ' end')
    return descriptions

def create_tokenizer(descriptions):
    lines = []
    for key in descriptions.keys():
        for desc in descriptions[key]:
            lines.append(desc)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = []
    for key in descriptions.keys():
        for desc in descriptions[key]:
            lines.append(desc)
    return max(len(d.split()) for d in lines)

def create_sequences(tokenizer, max_length, descriptions, features, vocab_size):
    """Create input-output sequence pairs for training"""
    X1, X2, y = [], [], []
    
    for key, desc_list in tqdm(descriptions.items(), desc="Creating sequences"):
        # Try both with and without .jpg extension
        feature_key = key + '.jpg' if key + '.jpg' in features else key
        if feature_key in features:
            feature = features[feature_key]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
    
    return np.array(X1), np.array(X2), np.array(y)

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model

def main():
    print("🚀 Simple Training: Image Caption Generation on 1000 Images")
    print("=" * 60)
    
    dataset_text = "Flickr8k_text"
    dataset_images = "Flickr8k_Dataset"
    
    # Step 1: Load pre-extracted features
    print("📁 Loading pre-extracted features...")
    features = load(open("features.p", "rb"))
    print(f"✓ Loaded features for {len(features)} images")
    
    # Step 2: Load training images (limit to 500 for faster training)
    train_path = os.path.join(dataset_text, 'Flickr_8k.trainImages.txt')
    all_train_imgs = load_photos(train_path, dataset_images)
    
    # Use even fewer images for faster training
    train_imgs = all_train_imgs[:500]
    print(f"📊 Using {len(train_imgs)} images for training (limited from {len(all_train_imgs)})")
    
    # Step 3: Load descriptions for these images
    train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
    print(f"📝 Loaded descriptions for {len(train_descriptions)} images")
    
    if len(train_descriptions) == 0:
        raise ValueError("No training descriptions found!")
    
    # Step 4: Create tokenizer and get vocabulary
    print("🔤 Creating tokenizer...")
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    max_desc_length = max_length(train_descriptions)
    
    print(f"✓ Vocabulary size: {vocab_size:,}")
    print(f"✓ Maximum description length: {max_desc_length}")
    
    # Step 5: Create training sequences
    print("📊 Creating training sequences...")
    X1, X2, y = create_sequences(tokenizer, max_desc_length, train_descriptions, features, vocab_size)
    
    print(f"✓ Created {len(X1):,} training sequences")
    print(f"✓ Feature input shape: {X1.shape}")
    print(f"✓ Text input shape: {X2.shape}")
    print(f"✓ Output shape: {y.shape}")
    
    # Step 6: Create and train model
    print("🧠 Creating model...")
    model = define_model(vocab_size, max_desc_length)
    
    # Train with fewer epochs for faster training
    epochs = 5
    batch_size = 32
    print(f"🏋️ Starting training for {epochs} epochs with batch size {batch_size}...")
    print("This should take about 5-10 minutes...")
    
    # Train the model
    history = model.fit([X1, X2], y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
    
    # Step 7: Save model and tokenizer
    print("💾 Saving model and tokenizer...")
    model.save('models/model_simple.h5')
    dump(tokenizer, open('models/tokenizer_simple.pkl', 'wb'))
    
    print("\n🎉 Training completed successfully!")
    print("=" * 60)
    print("📁 Files saved:")
    print("   - models/model_simple.h5 (trained model)")
    print("   - models/tokenizer_simple.pkl (tokenizer)")
    print("\n🧪 To test the model:")
    print("   python test_simple.py --image path/to/your/image.jpg")
    
    # Show training history
    print(f"\n📈 Final training loss: {history.history['loss'][-1]:.4f}")
    if 'val_loss' in history.history:
        print(f"📈 Final validation loss: {history.history['val_loss'][-1]:.4f}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Training script for FULL Flickr8k dataset (8,091 images)
This will take significantly longer but produce much better results
"""

import os
import string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pickle import dump, load
from tqdm import tqdm
import time

import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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
    
    print("Creating training sequences...")
    total_sequences = 0
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            total_sequences += len(desc.split()) - 1
    
    print(f"Expected total sequences: {total_sequences:,}")
    
    with tqdm(total=len(descriptions), desc="Processing images") as pbar:
        for key, desc_list in descriptions.items():
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
            pbar.update(1)
    
    return np.array(X1), np.array(X2), np.array(y)

def define_model(vocab_size, max_length):
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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    return model

def main():
    print("🚀 FULL DATASET TRAINING: Image Caption Generation on 8k Images")
    print("=" * 70)
    print("⚠️  WARNING: This will take 2-4 hours and use significant memory!")
    print("⚠️  Make sure you have at least 8GB RAM and 2GB free disk space")
    print("=" * 70)
    
    # Confirm user wants to proceed
    response = input("Do you want to proceed with full dataset training? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Training cancelled.")
        return
    
    dataset_text = "Flickr8k_text"
    dataset_images = "Flickr8k_Dataset"
    
    # Step 1: Load pre-extracted features
    print("\n📁 Loading pre-extracted features...")
    features = load(open("features.p", "rb"))
    print(f"✓ Loaded features for {len(features)} images")
    
    # Step 2: Load ALL training images (full dataset)
    train_path = os.path.join(dataset_text, 'Flickr_8k.trainImages.txt')
    all_train_imgs = load_photos(train_path, dataset_images)
    
    # Use ALL training images (6000 images)
    train_imgs = all_train_imgs  # No limit!
    print(f"📊 Using {len(train_imgs)} images for training (FULL DATASET)")
    
    # Step 3: Load descriptions for these images
    print("📝 Loading descriptions...")
    train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
    print(f"✓ Loaded descriptions for {len(train_descriptions)} images")
    
    if len(train_descriptions) == 0:
        raise ValueError("No training descriptions found!")
    
    # Step 4: Create tokenizer and get vocabulary
    print("🔤 Creating tokenizer...")
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    max_desc_length = max_length(train_descriptions)
    
    print(f"✓ Vocabulary size: {vocab_size:,}")
    print(f"✓ Maximum description length: {max_desc_length}")
    
    # Step 5: Create training sequences (this will take a while!)
    print("📊 Creating training sequences (this may take 10-20 minutes)...")
    start_time = time.time()
    
    X1, X2, y = create_sequences(tokenizer, max_desc_length, train_descriptions, features, vocab_size)
    
    sequence_time = time.time() - start_time
    print(f"✓ Created {len(X1):,} training sequences in {sequence_time:.1f} seconds")
    print(f"✓ Feature input shape: {X1.shape}")
    print(f"✓ Text input shape: {X2.shape}")
    print(f"✓ Output shape: {y.shape}")
    
    # Step 6: Create and train model
    print("🧠 Creating model...")
    model = define_model(vocab_size, max_desc_length)
    
    # Training configuration for full dataset
    epochs = 20  # More epochs for better results
    batch_size = 64  # Larger batch size for efficiency
    validation_split = 0.2
    
    print(f"🏋️ Training Configuration:")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Validation Split: {validation_split}")
    print(f"   - Training Samples: {int(len(X1) * (1 - validation_split)):,}")
    print(f"   - Validation Samples: {int(len(X1) * validation_split):,}")
    
    # Callbacks for better training
    callbacks = [
        ModelCheckpoint(
            'models/model_8k_best.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"\n🏋️ Starting training...")
    print("This will take 2-4 hours depending on your hardware...")
    print("You can monitor progress and stop early if needed (Ctrl+C)")
    
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        [X1, X2], y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=validation_split,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    
    # Step 7: Save final model and tokenizer
    print("💾 Saving final model and tokenizer...")
    model.save('models/model_8k_final.h5')
    dump(tokenizer, open('models/tokenizer_8k.pkl', 'wb'))
    
    # Save training history
    dump(history.history, open('models/training_history_8k.pkl', 'wb'))
    
    print("\n🎉 Training completed successfully!")
    print("=" * 70)
    print("📁 Files saved:")
    print("   - models/model_8k_best.h5 (best model during training)")
    print("   - models/model_8k_final.h5 (final model)")
    print("   - models/tokenizer_8k.pkl (tokenizer)")
    print("   - models/training_history_8k.pkl (training history)")
    
    print(f"\n📈 Training Results:")
    print(f"   - Training Time: {training_time/3600:.1f} hours")
    print(f"   - Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"   - Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    if 'val_loss' in history.history:
        print(f"   - Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"   - Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"   - Best Validation Loss: {min(history.history['val_loss']):.4f}")
    
    print("\n🧪 To test the new model:")
    print("   python test_8k.py --image path/to/your/image.jpg")
    print("\n📊 To compare with the simple model:")
    print("   python compare_models.py --image path/to/your/image.jpg")

if __name__ == '__main__':
    main()

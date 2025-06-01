#!/usr/bin/env python3
"""
Modern Web UI for Image Caption Generation
"""

import os
import sys
import json
import numpy as np
from pickle import load
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
models_cache = {}
xception_model = None

def extract_features(filename, model):
    """Extract features from an image using Xception model"""
    try:
        image = Image.open(filename)
    except Exception as e:
        return None, f"Couldn't open image: {e}"

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
    return feature, None

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

def load_model_and_tokenizer(model_type='simple'):
    """Load the trained model and tokenizer"""
    global models_cache, xception_model

    if model_type in models_cache:
        return models_cache[model_type]

    if model_type == 'simple':
        model_path = 'models/model_simple.h5'
        tokenizer_path = 'models/tokenizer_simple.pkl'
    elif model_type == '8k':
        model_path = 'models/model_8k_best.h5'
        tokenizer_path = 'models/tokenizer_8k.pkl'
    else:
        return None, None, None, None, "Invalid model type"

    if not os.path.exists(model_path):
        return None, None, None, None, f"Model not found at {model_path}"

    if not os.path.exists(tokenizer_path):
        return None, None, None, None, f"Tokenizer not found at {tokenizer_path}"

    try:
        # Load tokenizer
        tokenizer = load(open(tokenizer_path, "rb"))
        vocab_size = len(tokenizer.word_index) + 1
        max_length = 35

        # Create model architecture and load weights
        model = define_model(vocab_size, max_length)
        model.load_weights(model_path)

        # Load Xception model for feature extraction (only once)
        if xception_model is None:
            xception_model = Xception(include_top=False, pooling="avg")

        # Cache the model
        models_cache[model_type] = (model, tokenizer, xception_model, max_length)

        return model, tokenizer, xception_model, max_length, None

    except Exception as e:
        return None, None, None, None, f"Error loading model: {e}"

def get_available_models():
    """Get list of available models"""
    models = []

    # Check for simple model
    if os.path.exists('models/model_simple.h5') and os.path.exists('models/tokenizer_simple.pkl'):
        models.append({
            'id': 'simple',
            'name': 'Simple Model',
            'description': 'Trained on 500 images (~10 min training)',
            'vocab_size': '2,272 words'
        })

    # Check for 8K model
    if os.path.exists('models/model_8k_best.h5') and os.path.exists('models/tokenizer_8k.pkl'):
        models.append({
            'id': '8k',
            'name': '8K Model',
            'description': 'Trained on 6,000 images (2-4 hours training)',
            'vocab_size': '8,000+ words'
        })

    return models

@app.route('/')
def index():
    """Main page"""
    available_models = get_available_models()
    return render_template('index.html', models=available_models)

@app.route('/api/models')
def api_models():
    """API endpoint to get available models"""
    return jsonify(get_available_models())

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint to generate caption"""
    try:
        # Get model type from request
        model_type = request.form.get('model', 'simple')

        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Load model
        model, tokenizer, xception_model, max_length, error = load_model_and_tokenizer(model_type)
        if error:
            return jsonify({'error': error}), 500

        # Extract features and generate caption
        photo, error = extract_features(filepath, xception_model)
        if error:
            return jsonify({'error': error}), 500

        # Generate caption
        description = generate_desc(model, tokenizer, photo, max_length)
        description = description.replace('start ', '').replace(' end', '')

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'caption': description,
            'model': model_type,
            'vocab_size': len(tokenizer.word_index) + 1
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("🚀 Starting Image Caption Generation Web UI...")
    print("📍 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
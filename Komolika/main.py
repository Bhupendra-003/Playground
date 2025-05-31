import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import add, Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.xception import Xception

tqdm.pandas()

# --- Utility Functions ---

def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.strip().split('\n')
    descriptions = {}
    for caption in captions:
        img, cap = caption.split('\t')
        img = img.split('#')[0].strip()  # Keep image.jpg only
        descriptions.setdefault(img, []).append(cap)
    return descriptions

def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption = img_caption.replace("-", " ")
            desc = img_caption.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1 and word.isalpha()]
            captions[img][i] = ' '.join(desc)
    return captions

def text_vocabulary(descriptions):
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

def save_descriptions(descriptions, filename):
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = "\n".join(lines)
    with open(filename, "w") as file:
        file.write(data)

# --- Feature Extraction ---

def load_xception_model():
    return Xception(include_top=False, pooling='avg', weights='imagenet')

def extract_features(directory, model):
    features = {}
    for img in tqdm(os.listdir(directory)):
        if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_path = os.path.join(directory, img)
        image = Image.open(image_path).resize((299, 299))
        image = np.array(image)
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = np.expand_dims(image, axis=0).astype('float32')
        image = tf.keras.applications.xception.preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[img] = feature[0]
    return features

# --- Data Loading ---

def load_photos(filename, dataset_images_path):
    doc = load_doc(filename)
    photos = set()
    for line in doc.strip().split('\n'):
        image = line.strip()
        if os.path.exists(os.path.join(dataset_images_path, image)):
            photos.add(image)
    return photos

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = {}
    for line in doc.strip().split('\n'):
        tokens = line.split('\t')
        if len(tokens) != 2:
            continue
        image_id, image_desc = tokens
        if image_id in dataset:
            descriptions.setdefault(image_id, []).append(image_desc)
    return descriptions

def load_features(photos, features_path="features.p"):
    all_features = load(open(features_path, "rb"))
    return {k: all_features[k] for k in photos if k in all_features}

def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

# --- Data Preparation ---

def create_sequences(tokenizer, max_length, desc_list, feature, vocab_size):
    X1, X2, y = [], [], []
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

def data_generator(descriptions, features, tokenizer, max_length, vocab_size, batch_size=32):
    while True:
        X1, X2, y = [], [], []
        for key, desc_list in descriptions.items():
            feature = features[key]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, desc_list, feature, vocab_size)
            for i in range(len(input_image)):
                X1.append(input_image[i])
                X2.append(input_sequence[i])
                y.append(output_word[i])
                if len(X1) == batch_size:
                    yield ([np.array(X1), np.array(X2)], np.array(y))
                    X1, X2, y = [], [], []

def get_steps_per_epoch(descriptions, batch_size=32):
    total_sequences = 0
    for img_captions in descriptions.values():
        for caption in img_captions:
            total_sequences += len(caption.split()) - 1
    return max(1, total_sequences // batch_size)

# --- Model ---

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
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# --- Main ---

if __name__ == '__main__':
    dataset_text = "Flickr8k_text"
    dataset_images = "Flickr8k_Dataset"

    # Step 1: Build descriptions.txt if not done already
    filename = os.path.join(dataset_text, "Flickr8k.token.txt")
    descriptions = all_img_captions(filename)
    clean_descriptions = cleaning_text(descriptions)
    save_descriptions(clean_descriptions, "descriptions.txt")

    # Step 2: Extract features (run only once)
    # model_cnn = load_xception_model()
    # features = extract_features(dataset_images, model_cnn)
    # dump(features, open("features.p", "wb"))

    # Step 3: Load pre-extracted data
    features = load(open("features.p", "rb"))

    train_path = os.path.join(dataset_text, 'Flickr_8k.trainImages.txt')
    train_imgs = load_photos(train_path, dataset_images)
    train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)

    if len(train_descriptions) == 0:
        raise ValueError("No training descriptions found. Check 'descriptions.txt' and 'Flickr_8k.trainImages.txt' for matching image filenames.")

    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    max_desc_length = max_length(train_descriptions)

    model = define_model(vocab_size, max_desc_length)
    steps = get_steps_per_epoch(train_descriptions)
    train_generator = data_generator(train_descriptions, features, tokenizer, max_desc_length, vocab_size)

    model.fit(train_generator, epochs=20, steps_per_epoch=steps, verbose=1)
    model.save('model.h5')
    dump(tokenizer, open('tokenizer.pkl', 'wb'))

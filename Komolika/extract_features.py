import os
import pickle
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

def extract_features(image_dir):
    # Load pre-trained InceptionV3 model + remove top layer (classification layer)
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    features = {}

    for img_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_name)

        # Skip if not a file (ignore directories, etc.)
        if not os.path.isfile(img_path):
            continue

        # Load image with target size 299x299 for InceptionV3
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features
        feature_vector = model.predict(x)
        features[img_name] = feature_vector.flatten()

    # Save the features dictionary as features.p
    with open("features.p", "wb") as f:
        pickle.dump(features, f)

    print(f"Saved features for {len(features)} images to features.p")

if __name__ == "__main__":
    # Provide your image directory path here
   image_directory = "/home/komolika/Documents/Image-Captioner/Flickr8k_Dataset/Flicker8k_Dataset"
   extract_features(image_directory)

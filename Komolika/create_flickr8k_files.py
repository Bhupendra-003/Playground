#!/usr/bin/env python3
"""
Create the required Flickr8k text files from the existing descriptions.txt
"""

import os
import random
from collections import defaultdict

def create_flickr8k_token_file():
    """Create Flickr8k.token.txt from descriptions.txt"""
    print("Creating Flickr8k.token.txt...")
    
    # Read descriptions.txt and convert to the expected format
    with open('descriptions.txt', 'r') as f:
        lines = f.readlines()
    
    # Group by image and add caption numbers
    image_captions = defaultdict(list)
    for line in lines:
        line = line.strip()
        if '\t' in line:
            img_name, caption = line.split('\t', 1)
            image_captions[img_name].append(caption)
    
    # Write in the expected format: image.jpg#0, image.jpg#1, etc.
    token_lines = []
    for img_name, captions in image_captions.items():
        for i, caption in enumerate(captions):
            token_lines.append(f"{img_name}#{i}\t{caption}")
    
    os.makedirs('Flickr8k_text', exist_ok=True)
    with open('Flickr8k_text/Flickr8k.token.txt', 'w') as f:
        f.write('\n'.join(token_lines))
    
    print(f"✓ Created Flickr8k.token.txt with {len(token_lines)} captions")
    return list(image_captions.keys())

def create_split_files(all_images):
    """Create train/dev/test split files"""
    print("Creating train/dev/test split files...")
    
    # Shuffle images for random split
    random.seed(42)  # For reproducible splits
    shuffled_images = all_images.copy()
    random.shuffle(shuffled_images)
    
    total_images = len(shuffled_images)
    
    # Standard Flickr8k splits: 6000 train, 1000 dev, 1000 test
    train_size = 6000
    dev_size = 1000
    test_size = min(1000, total_images - train_size - dev_size)
    
    # Adjust if we don't have enough images
    if total_images < 8000:
        print(f"Warning: Only {total_images} images available, adjusting splits...")
        train_size = int(0.75 * total_images)  # 75% for training
        dev_size = int(0.125 * total_images)   # 12.5% for dev
        test_size = total_images - train_size - dev_size  # Rest for test
    
    train_images = shuffled_images[:train_size]
    dev_images = shuffled_images[train_size:train_size + dev_size]
    test_images = shuffled_images[train_size + dev_size:train_size + dev_size + test_size]
    
    # Write split files
    with open('Flickr8k_text/Flickr_8k.trainImages.txt', 'w') as f:
        f.write('\n'.join(train_images))
    
    with open('Flickr8k_text/Flickr_8k.devImages.txt', 'w') as f:
        f.write('\n'.join(dev_images))
    
    with open('Flickr8k_text/Flickr_8k.testImages.txt', 'w') as f:
        f.write('\n'.join(test_images))
    
    print(f"✓ Created split files:")
    print(f"  - Training: {len(train_images)} images")
    print(f"  - Development: {len(dev_images)} images") 
    print(f"  - Test: {len(test_images)} images")
    
    return train_images, dev_images, test_images

def verify_files():
    """Verify that all required files exist and have correct content"""
    print("\nVerifying created files...")
    
    required_files = [
        'Flickr8k_text/Flickr8k.token.txt',
        'Flickr8k_text/Flickr_8k.trainImages.txt',
        'Flickr8k_text/Flickr_8k.devImages.txt',
        'Flickr8k_text/Flickr_8k.testImages.txt'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            print(f"✓ {file_path}: {lines} lines")
        else:
            print(f"✗ {file_path}: Missing!")
    
    # Check if images in split files actually exist
    dataset_dir = 'Flickr8k_Dataset'
    if os.path.exists(dataset_dir):
        actual_images = set(os.listdir(dataset_dir))
        actual_images = {img for img in actual_images if img.lower().endswith('.jpg')}
        
        for split_file in ['Flickr_8k.trainImages.txt', 'Flickr_8k.devImages.txt', 'Flickr_8k.testImages.txt']:
            split_path = f'Flickr8k_text/{split_file}'
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    split_images = set(line.strip() for line in f.readlines())
                
                missing_images = split_images - actual_images
                if missing_images:
                    print(f"⚠️  {split_file}: {len(missing_images)} images not found in dataset")
                else:
                    print(f"✓ {split_file}: All images exist in dataset")

def main():
    print("🔧 Creating Flickr8k text files from existing data...")
    
    # Create token file and get list of all images
    all_images = create_flickr8k_token_file()
    
    # Create train/dev/test splits
    train_images, dev_images, test_images = create_split_files(all_images)
    
    # Verify everything was created correctly
    verify_files()
    
    print("\n🎉 All Flickr8k text files created successfully!")
    print("\nYou can now run the original training script:")
    print("  python main.py")

if __name__ == '__main__':
    main()

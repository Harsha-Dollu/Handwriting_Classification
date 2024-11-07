import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Function to augment images
def augment_images(images, augmentations=10):
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),              # Horizontal flips
        iaa.Affine(rotate=(-25, 25)), # Random rotations
        iaa.Multiply((0.8, 1.2)),     # Random brightness
        iaa.GaussianBlur(sigma=(0, 1.0)), # Blur images
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)) # Add noise
    ])
    augmented_images = []
    for image in images:
        for _ in range(augmentations):
            augmented_image = aug.augment_image(image)
            augmented_images.append(augmented_image)
    return augmented_images

# Function to save images to a folder
def save_images(images, folder):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(folder, f'image_{i}.png'), img)

def main():
    # Define paths
    good_folder = 'C:\\Users\\harsh\\OneDrive\\Desktop\\python\\python_project1\\goodHandwriting'
    bad_folder = 'C:\\Users\\harsh\\OneDrive\\Desktop\\python\\python_project1\\badHandwriting'
  
    
    augmented_good_folder = 'C:\\Users\\harsh\\OneDrive\\Desktop\\python\\python_project1\\augmentedGoodHandwriting'
    augmented_bad_folder = 'C:\\Users\\harsh\\OneDrive\\Desktop\\python\\python_project1\\augmentedBadHandwriting'
   
    good_images = load_images_from_folder(good_folder)
    bad_images = load_images_from_folder(bad_folder)
   
    augmented_good_images = augment_images(good_images, augmentations=9)
    augmented_bad_images = augment_images(bad_images, augmentations=14) # Heavier augmentation for balancing

    
    # Combine original and augmented images
    all_good_images = good_images + augmented_good_images
    all_bad_images = bad_images + augmented_bad_images

    
    # Save augmented images
    save_images(all_good_images, augmented_good_folder)
    save_images(all_bad_images, augmented_bad_folder)

    
    print("Augmentation completed and images saved.")

if __name__ == "__main__":
    main()

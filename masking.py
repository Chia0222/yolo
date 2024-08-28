import os
import cv2
import numpy as np

def apply_mask(image, mask_file):
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(mask_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        try:
            parts = line.strip().split()
            if len(parts) < 6:
                print(f"Skipping invalid line in {mask_file}: {line.strip()}")
                continue

            _, x_center, y_center, width, height, *mask_coords = map(float, parts)
            x_center *= w
            y_center *= h
            width *= w
            height *= h

            mask_coords = np.array(mask_coords).reshape(-1, 2)
            mask_coords[:, 0] *= w
            mask_coords[:, 1] *= h
            mask_coords = mask_coords.astype(np.int32)

            cv2.fillPoly(mask, [mask_coords], 255)

        except ValueError as e:
            print(f"Error processing line: {line.strip()} in {mask_file}: {e}")

    return mask

def process_images_and_masks(image_dir, mask_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_images = []
    
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(image_dir, image_name)
            mask_path = os.path.join(mask_dir, image_name.replace('.jpg', '.txt'))
            
            if not os.path.exists(mask_path):
                print(f"Mask file missing for image: {image_name}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue

            mask = apply_mask(image, mask_path)
            
            output_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_path, mask)
            
            processed_images.append(image_name)
    
    return processed_images

def validate_pairs(image_dir, mask_dir):
    images = {img for img in os.listdir(image_dir) if img.endswith('.jpg')}
    masks = {mask.replace('.txt', '.jpg') for mask in os.listdir(mask_dir) if mask.endswith('.txt')}
    
    missing_masks = images - masks
    missing_images = masks - images
    
    if missing_masks:
        print(f"Images without corresponding masks: {', '.join(missing_masks)}")
    if missing_images:
        print(f"Masks without corresponding images: {', '.join(missing_images)}")

# Directories
trainA_dir = 'bellpepper/trainA'
trainA_mask_dir = 'bellpepper/trainA_mask'
output_masked_trainA_dir = 'bellpepper/masked_trainA'

trainB_dir = 'bellpepper/trainB'
trainB_mask_dir = 'bellpepper/trainB_mask'
output_masked_trainB_dir = 'bellpepper/masked_trainB'

# Process the images and masks
print("Processing trainA images and masks...")
processed_trainA = process_images_and_masks(trainA_dir, trainA_mask_dir, output_masked_trainA_dir)
print(f"Processed {len(processed_trainA)} images in trainA.")

print("Processing trainB images and masks...")
processed_trainB = process_images_and_masks(trainB_dir, trainB_mask_dir, output_masked_trainB_dir)
print(f"Processed {len(processed_trainB)} images in trainB.")

# Validate image and mask pairs
print("Validating trainA image and mask pairs...")
validate_pairs(trainA_dir, trainA_mask_dir)

print("Validating trainB image and mask pairs...")
validate_pairs(trainB_dir, trainB_mask_dir)

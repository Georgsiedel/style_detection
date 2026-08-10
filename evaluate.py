import os
import cv2
import yaml
import argparse
from pathlib import Path
from imagecorruptions import corrupt, get_corruption_names
from ultralytics import YOLO, RTDETR

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Model Robustness on Hendrycks Corruptions")
    parser.add_argument("--model_path", type=str, default="runs/detect/train/weights/best.pt", help="Path to the trained weights file")
    parser.add_argument("--model_type", type=str, default="yolo26", choices=["yolo26", "rtdetr"], help="Architecture used")
    parser.add_argument("--clean_yaml", type=str, default="../datasets/coco8.yaml", help="Path to original dataset YAML")
    parser.add_argument("--dataset", type=str, default="coco8", help="Name for the cache folder (e.g., coco8, voc, coco)")
    parser.add_argument("--val_images_dir", type=str, default="../datasets/coco8/images/val", help="Path to the clean validation images folder")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Infer the original labels folder by replacing 'images' with 'labels' in the path
    val_labels_dir = args.val_images_dir.replace('images', 'labels')
    if not os.path.exists(val_labels_dir):
        print(f"Warning: Original labels directory not found at {val_labels_dir}. Metrics will fail.")
    abs_val_labels_dir = os.path.abspath(val_labels_dir)

    # Base directory for permanently caching corrupted datasets
    cache_base_dir = os.path.abspath(f"../datasets/{args.dataset}_corrupted_cache")
    os.makedirs(cache_base_dir, exist_ok=True)

    # Load trained model
    if args.model_type == "yolo26":
        model = YOLO(args.model_path)
    else:
        model = RTDETR(args.model_path)

    # Evaluate on CLEAN data first
    print(f"\n--- Evaluating {args.model_path} on Clean Data ---")
    clean_metrics = model.val(data=args.clean_yaml, verbose=False)
    clean_map = clean_metrics.box.map
    print(f"Clean mAP50-95: {clean_map:.4f}\n")

    all_corruptions = get_corruption_names('common') + get_corruption_names('validation')
    severities = [1, 2, 3, 4, 5]
    
    total_corrupt_map = 0.0
    evaluation_count = 0

    print("--- Starting Robustness Benchmarking ---")
    
    for corruption in all_corruptions:
        for severity in severities:
            # Setup specific cache folder for this corruption and severity
            target_dir = os.path.join(cache_base_dir, f"{corruption}_{severity}")
            
            # Ultralytics strictly requires an 'images' and 'labels' subfolder structure
            target_images_dir = os.path.join(target_dir, "images", "val")
            target_labels_dir = os.path.join(target_dir, "labels")
            target_labels_val_dir = os.path.join(target_labels_dir, "val")
            
            os.makedirs(target_images_dir, exist_ok=True)
            os.makedirs(target_labels_dir, exist_ok=True)
            
            # Create a symlink to the original labels so YOLO can find them effortlessly
            if not os.path.exists(target_labels_val_dir):
                os.symlink(abs_val_labels_dir, target_labels_val_dir)
            
            clean_images = [f for f in os.listdir(args.val_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Check if this corruption dataset has already been generated
            if len(os.listdir(target_images_dir)) < len(clean_images):
                print(f"Generating cache for {corruption} (Severity {severity})...")
                for img_name in clean_images:
                    img_path = os.path.join(args.val_images_dir, img_name)
                    image = cv2.imread(img_path)
                    
                    if image is not None:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        corrupted_rgb = corrupt(image_rgb, corruption_name=corruption, severity=severity)
                        corrupted_bgr = cv2.cvtColor(corrupted_rgb, cv2.COLOR_RGB2BGR)
                        # Write the image to the inner 'images/val' folder
                        cv2.imwrite(os.path.join(target_images_dir, img_name), corrupted_bgr)
            
            # Dynamically generate a YAML file for this specific cached folder
            temp_yaml_path = os.path.join(cache_base_dir, "temp_eval.yaml")
            yaml_data = {
                'path': target_dir,
                'train': '', # Not needed for validation
                'val': 'images/val',  # Point to the newly structured images subfolder
                'names': model.names
            }
            with open(temp_yaml_path, "w") as f:
                yaml.dump(yaml_data, f, sort_keys=False)

            # Validate against the cached corrupted dataset
            metrics = model.val(data=temp_yaml_path, verbose=False)
            current_map = metrics.box.map
            
            print(f"Corruption: {corruption.ljust(20)} | Severity: {severity} | mAP50-95: {current_map:.4f}")
            
            total_corrupt_map += current_map
            evaluation_count += 1

    average_robust_map = total_corrupt_map / evaluation_count
    
    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    print(f"Weights:                     {args.model_path}")
    print(f"Dataset:                     {args.dataset}")
    print(f"Test mAP (Clean Data):       {clean_map:.4f}")
    print(f"Robust mAP (19 Corruptions): {average_robust_map:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
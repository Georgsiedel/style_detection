import os
import cv2
import yaml
import argparse
import csv
from pathlib import Path
import numpy as np

# monkey patch - imagecorruptions uses legacy NumPy aliases removed in NumPy 2.0
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64

from imagecorruptions import corrupt, get_corruption_names
import imagecorruptions
from ultralytics import YOLO, RTDETR

# monkey patch - imagecorruptions glass blur uses triple for-loop over image dims, which is too heavy for 640px voc and coco images
def fast_glass_blur(x, severity=1):
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    
    sigma = c[0]
    max_delta = c[1]
    iterations = c[2]
    
    # skimage's gaussian blur truncates at 4.0 standard deviations by default. 
    # We calculate the equivalent cv2 kernel size to match skimage exactly.
    ksize = int(2 * np.ceil(4.0 * sigma) + 1)
    
    # 1. First Blur Pass
    x_blurred = cv2.GaussianBlur(np.array(x), (ksize, ksize), sigma)
    
    # 2. Vectorized Pixel Shuffle
    h, w = x_blurred.shape[:2]
    for _ in range(iterations):
        # Original legacy code uses np.random.randint(-c, c) which excludes +c
        dy = np.random.randint(-max_delta, max_delta, size=(h, w))
        dx = np.random.randint(-max_delta, max_delta, size=(h, w))
        y, x_idx = np.mgrid[0:h, 0:w]
        
        y_prime = np.clip(y + dy, 0, h - 1)
        x_prime = np.clip(x_idx + dx, 0, w - 1)
        
        x_blurred = x_blurred[y_prime, x_prime]
        
    # 3. Second Blur Pass (Missing in the previous patch!)
    x_final = cv2.GaussianBlur(x_blurred, (ksize, ksize), sigma)
    
    return x_final

# Inject fast version into the package before the benchmark starts
imagecorruptions.corruption_dict['glass_blur'] = fast_glass_blur

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Model Robustness on Hendrycks Corruptions")
    parser.add_argument("--model_path", type=str, default="runs/detect/train/weights/best.pt", help="Path to the trained weights file")
    parser.add_argument("--model_type", type=str, default="yolo26", choices=["yolo26", "rtdetr"], help="Architecture used")
    parser.add_argument("--clean_yaml", type=str, default="../datasets/coco8.yaml", help="Path to original dataset YAML")
    parser.add_argument("--dataset", type=str, default="coco8", help="Name for the cache folder (e.g., coco8, voc, coco)")
    parser.add_argument("--val_images_dir", type=str, default="../datasets/coco8/images/val", help="Path to the clean validation images folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size. Note: RT-DETR usually requires a smaller batch (e.g. 4) than YOLO.")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Infer the original labels folder by replacing 'images' with 'labels' in the path
    val_labels_dir = args.val_images_dir.replace('images', 'labels')
    if not os.path.exists(val_labels_dir):
        print(f"Warning: Original labels directory not found at {val_labels_dir}. Metrics will fail.")
    abs_val_labels_dir = os.path.abspath(val_labels_dir)

    # Setup the master evaluation folder and CSV
    model_path_obj = Path(args.model_path)
    
    # Check if the path follows the standard structure (e.g., runs/detect/name/weights/best.pt)
    if len(model_path_obj.parts) >= 4:
        train_folder_name = model_path_obj.parents[1].name
    else:
        train_folder_name = model_path_obj.stem
        
    # Switch "train-" to "val-" for the output folder name
    if train_folder_name.startswith("train-"):
        val_folder_name = train_folder_name.replace("train-", "val-", 1)
    else:
        val_folder_name = f"val-{train_folder_name}"
        
    # Construct the base directory: runs/detect/{dataset}_{model_type}
    base_runs_dir = Path("runs/detect") / f"{args.dataset}_{args.model_type}"
    eval_project_dir = os.path.abspath(base_runs_dir / val_folder_name)
    os.makedirs(eval_project_dir, exist_ok=True)
    
    csv_file_path = os.path.join(eval_project_dir, "robustness_metrics.csv")
    csv_data = [["Condition", "Severity", "mAP50", "mAP50-95"]]

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
    
    # Override the default save location to our new custom master directory
    clean_metrics = model.val(data=args.clean_yaml, project=eval_project_dir, name="clean", verbose=False, batch=args.batch_size)
    
    # Extract the exact metrics programmatically 
    clean_map50 = clean_metrics.box.map50
    clean_map = clean_metrics.box.map
    print(f"Clean mAP50-95: {clean_map:.4f}\n")
    
    csv_data.append(["Clean", 0, f"{clean_map50:.4f}", f"{clean_map:.4f}"])
    
    # Save CSV immediately
    with open(csv_file_path, mode="w", newline="") as f:
        csv.writer(f).writerows(csv_data)

    all_corruptions = get_corruption_names('common') + get_corruption_names('validation')
    severities = [1, 2, 3, 4, 5]
    
    total_corrupt_map = 0.0
    total_corrupt_map50 = 0.0
    evaluation_count = 0

    print("--- Starting Robustness Benchmarking ---")
    
    for corruption in all_corruptions:
        for severity in severities:
            # Setup specific cache folder for this corruption and severity
            target_dir = os.path.join(cache_base_dir, f"{corruption}_{severity}")
            
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
                        cv2.imwrite(os.path.join(target_images_dir, img_name), corrupted_bgr)
            
            # Dynamically generate a YAML file for this specific cached folder
            temp_yaml_path = os.path.join(cache_base_dir, "temp_eval.yaml")
            yaml_data = {
                'path': target_dir,
                'train': '', 
                'val': 'images/val',  
                'names': model.names
            }
            with open(temp_yaml_path, "w") as f:
                yaml.dump(yaml_data, f, sort_keys=False)

            # Validate against the cached corrupted dataset, saving results into a specific subfolder
            run_name = f"{corruption}_{severity}"
            metrics = model.val(data=temp_yaml_path, project=eval_project_dir, name=run_name, verbose=False, batch=args.batch_size)
            
            current_map50 = metrics.box.map50
            current_map = metrics.box.map
            
            print(f"Corruption: {corruption.ljust(20)} | Severity: {severity} | mAP50: {current_map50:.4f} | mAP50-95: {current_map:.4f}")
            
            # Append metrics and update the CSV instantly
            csv_data.append([corruption, severity, f"{current_map50:.4f}", f"{current_map:.4f}"])
            with open(csv_file_path, mode="w", newline="") as f:
                csv.writer(f).writerows(csv_data)
            
            total_corrupt_map += current_map
            total_corrupt_map50 += current_map50
            evaluation_count += 1

    average_robust_map = total_corrupt_map / evaluation_count
    average_robust_map_50 = total_corrupt_map50 / evaluation_count
    
    # Append the final average score to the CSV
    csv_data.append(["AVERAGE_ROBUSTNESS", "-", f"{average_robust_map_50:.4f}", f"{average_robust_map:.4f}"])
    with open(csv_file_path, mode="w", newline="") as f:
        csv.writer(f).writerows(csv_data)
    
    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    print(f"Results Directory:           {eval_project_dir}")
    print(f"Test mAP (Clean Data):       {clean_map:.4f}")
    print(f"Robust mAP (19 Corruptions): {average_robust_map:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
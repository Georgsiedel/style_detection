import argparse
from ultralytics import YOLO, RTDETR
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.rtdetr.train import RTDETRTrainer
from custom_trainers import get_custom_trainer
from pathlib import Path
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train Object Detectors with Batched GPU Style Augmentation")
    parser.add_argument("--model_type", type=str, default="yolo26", choices=["yolo26", "rtdetr"], help="Model architecture")
    parser.add_argument("--style_type", type=str, default="none", choices=["ast", "nst", "none"], help="Type of style augmentation")
    parser.add_argument("--prob", type=float, default=0.0, help="Probability of applying style transfer if style_type is not none")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset name (e.g., voc, coco, coco8)")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Note: RT-DETR usually requires a smaller batch (e.g. 4) than YOLO.")
    parser.add_argument("--finetune", action="store_true", help="Flag to finetune pretrained models. If omitted, trains from scratch.")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--weights", type=str, default="", help="Path to last.pt to resume from")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Dataset Mapping Logic ---
    # Clean the input to lowercase and strip .yaml if the user accidentally typed it
    clean_dataset_name = args.dataset.lower().replace(".yaml", "")
    
    # Map the clean string to the exact case-sensitive filename Ultralytics expects
    yaml_mapping = {
        "voc": "VOC.yaml",
        "coco": "coco.yaml",
        "coco8": "coco8.yaml"
    }
    dataset_yaml = yaml_mapping.get(clean_dataset_name, f"{clean_dataset_name}.yaml")

    # 1. Setup Architecture
    if args.resume:
        # When resuming, load the last.pt checkpoint directly
        if not args.weights:
            raise ValueError("You must provide --weights path/to/last.pt when using --resume")
        model = YOLO(args.weights) if args.model_type == "yolo26" else RTDETR(args.weights)
        base_trainer = DetectionTrainer if args.model_type == "yolo26" else RTDETRTrainer
    else:
        if args.model_type == "yolo26":
            model_source = "yolo26s-objv1-150.pt" if args.finetune else "yolo26s.yaml"
            model = YOLO(model_source) 
            base_trainer = DetectionTrainer
        elif args.model_type == "rtdetr":
            model_source = "rtdetr-l.pt" if args.finetune else "rtdetr-l.yaml"
            model = RTDETR(model_source) 
            base_trainer = RTDETRTrainer

    # 2. Generate Custom Trainer with parameters
    # We must generate this even when resuming so the augmentations are preserved
    CustomTrainerClass = get_custom_trainer(
        base_trainer_class=base_trainer, 
        style_type=args.style_type, 
        probability=args.prob
    )

    # 3. Configure Training Logic
    if args.resume:
        # When resuming, Ultralytics loads epochs, optimizer, and dataset from the checkpoint
        train_kwargs = {
            "resume": True,
            "trainer": CustomTrainerClass
        }
        print(f"\nResuming {args.model_type.upper()} from {args.weights}")

    else:
        finetune_tag = '_finetune' if args.finetune else ''
        train_kwargs = {
            "data": dataset_yaml,
            "epochs": args.epochs,
            "imgsz": 640,
            "batch": args.batch_size,
            "device": 0,
            "pretrained": args.finetune,
            "trainer": CustomTrainerClass,
            "project": f"{clean_dataset_name}_{args.model_type}",
            "name": f"train-{args.style_type}_{args.prob}_{args.epochs}epochs{finetune_tag}"
        }

        if args.finetune:
            train_kwargs["optimizer"] = "AdamW" if args.model_type == "rtdetr" else "MuSGD"
            train_kwargs["lr0"] = 1e-5 if args.model_type == "rtdetr" else 3.8e-4
            train_kwargs["warmup_epochs"] = 0.99
        else:
            train_kwargs["optimizer"] = "auto"

        print(f"\nStarting {args.model_type.upper()} | Finetune: {args.finetune} | Optim: {train_kwargs['optimizer']} | Batch: {args.batch_size}")
    
    print(f"Style: {args.style_type.upper()} (p={args.prob})\n")

    # 4. Start Training
    model.train(**train_kwargs)
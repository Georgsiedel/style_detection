import argparse
from ultralytics import YOLO, RTDETR
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.rtdetr.train import RTDETRTrainer
from custom_trainers import get_custom_trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Object Detectors with Batched GPU Style Augmentation")
    parser.add_argument("--model_type", type=str, default="yolo26", choices=["yolo26", "rtdetr"], help="Model architecture")
    parser.add_argument("--style_type", type=str, default="none", choices=["ast", "nst", "none"], help="Type of style augmentation")
    parser.add_argument("--prob", type=float, default=0.2, help="Probability of applying style transfer if style_type is not none")
    parser.add_argument("--dataset", type=str, default="/home/siedel/datasets/coco8.yaml", help="Path to dataset YAML (e.g., VOC.yaml or coco8.yaml)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Note: RT-DETR usually requires a smaller batch (e.g. 4) than YOLO.")
    parser.add_argument("--finetune", action="store_true", help="Flag to finetune pretrained models. If omitted, trains from scratch.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1. Setup Architecture
    if args.model_type == "yolo26":
        # Load .pt for finetuning, .yaml for random initialization (from scratch)
        model_source = "yolo26s.pt" if args.finetune else "yolo26s.yaml"
        model = YOLO(model_source) 
        base_trainer = DetectionTrainer
        
    elif args.model_type == "rtdetr":
        model_source = "rtdetr-l.pt" if args.finetune else "rtdetr-l.yaml"
        model = RTDETR(model_source) 
        base_trainer = RTDETRTrainer

    # 2. Configure Training Logic
    train_kwargs = {
        "data": args.dataset,
        "epochs": args.epochs,
        "imgsz": 640,
        "batch": args.batch_size,
        "device": 0,
        "pretrained": args.finetune
    }

    if args.finetune:
        # --- FINE-TUNING MODE ---
        # Force specific optimizers to disable Ultralytics' "auto" override
        train_kwargs["optimizer"] = "AdamW" if args.model_type == "rtdetr" else "SGD"
        # Set explicitly low learning rates to protect pretrained features
        train_kwargs["lr0"] = 1e-5 if args.model_type == "rtdetr" else 1e-4
        # Disable warmup since the weights are already stable
        train_kwargs["warmup_epochs"] = 0.0
    else:
        # --- FROM SCRATCH MODE ---
        # Leave optimizer on "auto" so Ultralytics calculates the best LR and uses standard warmup
        train_kwargs["optimizer"] = "auto"

    # 3. Generate Custom Trainer with parameters
    CustomTrainerClass = get_custom_trainer(
        base_trainer_class=base_trainer, 
        style_type=args.style_type, 
        probability=args.prob
    )
    
    train_kwargs["trainer"] = CustomTrainerClass

    # 4. Start Training
    print(f"Starting {args.model_type.upper()} | Finetune: {args.finetune} | Optim: {train_kwargs['optimizer']} | Batch: {args.batch_size} | Style: {args.style_type.upper()} (p={args.prob})")
    
    model.train(**train_kwargs)
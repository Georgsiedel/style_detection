import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.rtdetr.train import RTDETRTrainer
from micro_ast_augmentation import MicroASTAugmentation
from nst_augmentation import NSTRectangularTransform

def get_custom_trainer(base_trainer_class, style_type: str, probability: float):
    """
    Class factory that returns a custom trainer class inheriting from the provided base_trainer_class.
    """
    class CustomDynamicTrainer(base_trainer_class):
        def preprocess_batch(self, batch: dict) -> dict:
            # 1. Standard Ultralytics preprocessing (moves to GPU, converts to float [0, 1])
            batch = super().preprocess_batch(batch)
            imgs = batch["img"] 
            
            # If no styling is requested, return immediately
            if style_type.lower() == "none" or probability <= 0.0:
                return batch
                
            # 2. Lazy initialization of the augmentation model on the correct device
            if not hasattr(self, "style_aug"):
                if style_type.lower() == "ast":
                    self.style_aug = MicroASTAugmentation(
                        style_feats_path="../datasets/style_distribution.npz",
                        content_encoder_path="MicroAST/models/style_encoder_iter_160000.pth.tar",
                        decoder_path="MicroAST/models/decoder_iter_160000.pth.tar",
                        device=imgs.device,
                        probability=probability
                    )
                elif style_type.lower() == "nst":
                    self.style_aug = NSTRectangularTransform.from_files(
                        style_feats_path="../datasets/style_feats_adain_1000.npy",
                        encoder_path="nst/vgg_normalised.pth",
                        decoder_path="nst/decoder.pth",
                        probability=probability
                    ).to(imgs.device)
                else:
                    raise ValueError(f"Unknown style_type: {style_type}")

            # 3. Apply the augmentation
            batch["img"] = self.style_aug(imgs)
            return batch
            
    return CustomDynamicTrainer
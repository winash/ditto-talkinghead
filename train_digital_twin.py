#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a personalized digital twin from a video of a person speaking.
This allows you to create a digital twin similar to Heygen.
"""

import os
import argparse
import time
from stream_pipeline_offline import StreamSDK
from core.train.personalization import DigitalTwinTrainer

def main():
    parser = argparse.ArgumentParser(description="Train a personalized digital twin from video")
    
    # Required parameters
    parser.add_argument("--source_video", type=str, required=True,
                       help="Path to video of person speaking")
    parser.add_argument("--output_dir", type=str, default="./digital_twin_model",
                       help="Directory to save trained model")
    
    # Model configuration
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus", 
                       help="Path to model data root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", 
                       help="Path to config pickle file")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--max_frames", type=int, default=500,
                       help="Maximum number of frames to extract from video")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training (cuda or cpu)")
    
    args = parser.parse_args()
    
    print("\n======================================")
    print("Ditto TalkingHead - Digital Twin Trainer")
    print("======================================")
    
    # Record start time
    start_time = time.time()
    
    # Initialize SDK
    print(f"Initializing SDK with models from {args.data_root}")
    sdk = StreamSDK(args.cfg_pkl, args.data_root)
    
    # Create trainer
    trainer = DigitalTwinTrainer(
        sdk=sdk,
        source_video_path=args.source_video,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run training
    print(f"\nStarting digital twin training from {args.source_video}")
    print(f"This will extract your facial expressions and speaking style")
    print(f"Training for {args.epochs} epochs with learning rate {args.learning_rate}")
    print(f"Results will be saved to {args.output_dir}")
    print("\nTraining may take a while - please be patient...")
    print("======================================\n")
    
    model_path = trainer.train()
    
    # Print performance stats
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n======================================")
    print(f"Digital twin training completed in {hours}h {minutes}m {seconds}s")
    print(f"Model saved to: {model_path}")
    print("\nTo use your digital twin, run inference with:")
    print(f"  --digital_twin_mode --digital_twin_model_dir {args.output_dir}")
    print("======================================")

if __name__ == "__main__":
    main()
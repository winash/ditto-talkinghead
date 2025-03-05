import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from tqdm import tqdm

class DigitalTwinTrainer:
    def __init__(
        self,
        sdk,
        source_video_path,
        source_audio_path=None,
        learning_rate=1e-5,
        epochs=50,
        output_dir="./digital_twin_model",
        device="cuda",
    ):
        """
        Trains a personalized digital twin from video of a person speaking.
        
        Args:
            sdk: The main StreamSDK instance
            source_video_path: Path to video of the person speaking
            source_audio_path: Optional separate audio path (if None, uses video audio)
            learning_rate: Learning rate for fine-tuning
            epochs: Number of training epochs
            output_dir: Directory to save the trained model
            device: Device to use for training ("cuda" or "cpu")
        """
        self.sdk = sdk
        self.source_video_path = source_video_path
        self.source_audio_path = source_audio_path
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the core components from SDK
        self.source2info = sdk.avatar_registrar.source2info
        self.lmdm = sdk.audio2motion.lmdm
        self.wav2feat = sdk.wav2feat
        
        # Initialize personal style parameters
        self.personal_style = {
            "expression_bias": None,  # Bias towards personal facial expressions
            "motion_scale": None,     # Scale factors for motion dimensions
            "emotion_mapping": None,  # Custom emotion mapping
            "style_embedding": None,  # Style embedding vector
        }
        
    def extract_frames(self, video_path, max_frames=500):
        """Extract frames and audio from video"""
        import librosa
        
        # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        
        # Progress bar for frame extraction
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_extract = min(total_frames, max_frames)
        pbar = tqdm(total=frames_to_extract, desc="Extracting frames")
        
        # Extract frames
        frame_count = 0
        while frame_count < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        
        # Extract audio if not provided separately
        if self.source_audio_path is None:
            import subprocess
            temp_audio_path = os.path.join(self.output_dir, "temp_audio.wav")
            
            # Extract audio from video
            cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_audio_path}"'
            subprocess.call(cmd, shell=True)
            
            # Load audio file
            audio, sr = librosa.load(temp_audio_path, sr=16000)
            self.source_audio_path = temp_audio_path
        else:
            audio, sr = librosa.load(self.source_audio_path, sr=16000)
            
        return frames, audio, sr, fps
    
    def extract_motion_features(self, frames):
        """Extract motion features from frames"""
        print("Extracting motion features from frames...")
        
        # Initialize feature extraction
        motion_features = []
        last_lmk = None
        
        # Extract features from frames with progress bar
        pbar = tqdm(total=len(frames), desc="Processing frames")
        
        for frame in frames:
            # Extract source info and motion features
            source_info = self.source2info(frame, last_lmk=last_lmk)
            
            # Update last landmark for tracking
            last_lmk = source_info["lmk203"]
            
            # Extract x_s_info (motion parameters)
            x_s_info = source_info["x_s_info"]
            
            # Convert to 1D feature vector like in _cvt_LP_motion_info
            motion_feature = np.concatenate([
                x_s_info["scale"].reshape(-1),
                x_s_info["pitch"].reshape(-1),
                x_s_info["yaw"].reshape(-1),
                x_s_info["roll"].reshape(-1),
                x_s_info["t"].reshape(-1),
                x_s_info["exp"].reshape(-1),
            ], axis=0)
            
            motion_features.append(motion_feature)
            pbar.update(1)
            
        pbar.close()
        
        # Convert to numpy array
        motion_features = np.array(motion_features)
        print(f"Extracted {len(motion_features)} motion features with shape {motion_features.shape}")
        
        return motion_features
    
    def extract_audio_features(self, audio, sr):
        """Extract audio features from audio"""
        print("Extracting audio features...")
        
        # Use the SDK's wav2feat to extract audio features
        audio_features = self.wav2feat.wav2feat(audio, sr=sr)
        
        print(f"Extracted audio features with shape {audio_features.shape}")
        return audio_features
    
    def analyze_personal_style(self, motion_features):
        """Analyze personal style from motion features"""
        print("Analyzing personal style...")
        
        # Calculate statistics of motion features
        mean_features = np.mean(motion_features, axis=0)
        std_features = np.std(motion_features, axis=0)
        
        # Extract expression component (based on dimension mapping in the system)
        exp_start = 197  # Based on the system's feature mapping
        exp_end = exp_start + 63
        
        # Calculate expression bias (how this person's expressions differ from average)
        expression_bias = mean_features[exp_start:exp_end]
        
        # Calculate motion scale factors (how dynamic each dimension is)
        # Use relative standard deviation normalized to typical values
        base_std = np.ones_like(std_features) * 0.01  # Baseline std values
        motion_scale = np.clip(std_features / base_std, 0.5, 2.0)
        
        # Save personal style parameters
        self.personal_style["expression_bias"] = expression_bias
        self.personal_style["motion_scale"] = motion_scale
        
        # Create a style embedding by concatenating key statistics
        style_embedding = np.concatenate([
            mean_features,
            std_features,
            np.quantile(motion_features, 0.25, axis=0),
            np.quantile(motion_features, 0.75, axis=0)
        ])
        
        self.personal_style["style_embedding"] = style_embedding
        
        print("Personal style analysis complete")
        return self.personal_style
    
    def finetune_model(self, motion_features, audio_features):
        """Fine-tune the LMDM model on the personal data"""
        print("Preparing for model fine-tuning...")
        
        # Extract LMDM model - this depends on whether it's PyTorch or another format
        # For this example, we'll focus on PyTorch model fine-tuning
        if self.lmdm.model_type != "pytorch":
            print("Fine-tuning requires PyTorch model. Current model type is:", self.lmdm.model_type)
            print("Creating a PyTorch adaptation model instead...")
            self._create_adaptation_model()
            return
            
        model = self.lmdm.model
        
        # Set model to training mode and prepare optimizer
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        
        # Prepare dataset from extracted features
        seq_frames = self.lmdm.seq_frames
        
        # Create sequences of motion and audio features
        sequences = []
        for i in range(0, len(motion_features) - seq_frames, seq_frames // 2):
            motion_seq = motion_features[i:i+seq_frames]
            audio_seq = audio_features[i:i+seq_frames]
            
            if len(motion_seq) == seq_frames and len(audio_seq) == seq_frames:
                sequences.append((motion_seq, audio_seq))
        
        print(f"Created {len(sequences)} training sequences")
        
        # Training loop
        print("Starting fine-tuning...")
        for epoch in range(self.epochs):
            epoch_loss = 0
            pbar = tqdm(sequences, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for motion_seq, audio_seq in pbar:
                # Prepare inputs
                motion_tensor = torch.tensor(motion_seq, dtype=torch.float32).to(self.device)
                audio_tensor = torch.tensor(audio_seq, dtype=torch.float32).to(self.device)
                
                # Forward pass
                pred = model.forward_train(motion_tensor[None], audio_tensor[None])
                loss = F.mse_loss(pred, motion_tensor[None])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{epoch_loss/(pbar.n+1):.6f}"})
            
            print(f"Epoch {epoch+1} loss: {epoch_loss/len(sequences):.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == self.epochs - 1:
                checkpoint_path = os.path.join(self.output_dir, f"lmdm_personalized_epoch{epoch+1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
        print("Fine-tuning complete")
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "lmdm_personalized_final.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved personalized model to {final_model_path}")
        
        # Save personal style parameters
        style_path = os.path.join(self.output_dir, "personal_style.npy")
        np.save(style_path, self.personal_style)
        
        return final_model_path
    
    def _create_adaptation_model(self):
        """Create an adaptation model when fine-tuning the main model is not possible"""
        # This would create a small neural network that adapts the output of the original model
        # Implementation depends on the specific architecture
        print("Adaptation model creation not implemented for non-PyTorch models yet")
        
        # For now, just save the personal style parameters
        style_path = os.path.join(self.output_dir, "personal_style.npy")
        np.save(style_path, self.personal_style)
        
    def train(self):
        """Run the full training pipeline"""
        print(f"Starting digital twin training from {self.source_video_path}")
        
        # Extract frames and audio
        frames, audio, sr, fps = self.extract_frames(self.source_video_path)
        
        # Extract motion features
        motion_features = self.extract_motion_features(frames)
        
        # Extract audio features
        audio_features = self.extract_audio_features(audio, sr)
        
        # Analyze personal style
        self.analyze_personal_style(motion_features)
        
        # Fine-tune model
        model_path = self.finetune_model(motion_features, audio_features)
        
        print(f"Digital twin training complete. Model saved to {model_path}")
        return model_path
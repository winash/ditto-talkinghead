import librosa
import math
import os
import numpy as np
import random
import torch
import pickle

from stream_pipeline_offline import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict = {}):

    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    # Extract digital twin and background motion parameters from setup_kwargs
    # to make them accessible to the StreamSDK instance
    SDK.digital_twin_mode = setup_kwargs.get("digital_twin_mode", False)
    SDK.digital_twin_model_dir = setup_kwargs.get("digital_twin_model_dir", None)
    SDK.bg_motion_enabled = setup_kwargs.get("bg_motion_enabled", True)
    SDK.bg_motion_intensity = setup_kwargs.get("bg_motion_intensity", 0.005)
    
    # Setup the SDK with all parameters
    SDK.setup(source_path, output_path, **setup_kwargs)
    
    # Load digital twin model if specified and enabled
    if SDK.digital_twin_mode and SDK.digital_twin_model_dir:
        # Check if method exists before trying to call it
        if hasattr(SDK, 'load_digital_twin'):
            SDK.load_digital_twin(SDK.digital_twin_model_dir)
        else:
            print("Warning: Digital twin mode enabled but SDK.load_digital_twin method not found")

    audio, sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80  # 6480
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i:i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
    SDK.close()

    cmd = f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    print(cmd)
    os.system(cmd)

    print(output_path)


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./checkpoints/ditto_trt_Ampere_Plus", help="path to trt data_root")
    parser.add_argument("--cfg_pkl", type=str, default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl", help="path to cfg_pkl")

    parser.add_argument("--audio_path", type=str, help="path to input wav")
    parser.add_argument("--source_path", type=str, help="path to input image")
    parser.add_argument("--output_path", type=str, help="path to output mp4")
    
    # Enhanced emotional expression parameters
    parser.add_argument("--emotional_intensity", type=float, default=1.3, 
                       help="Controls emotional expressiveness (1.0=normal, 1.3=enhanced, 1.5=strong)")
    parser.add_argument("--sampling_timesteps", type=int, default=50,
                       help="Number of sampling steps (50-80, higher is more detailed)")
    parser.add_argument("--noise_guidance", type=float, default=0.3,
                       help="Noise guidance for more expressive motion (0.1-0.3)")
    parser.add_argument("--smo_k_d", type=int, default=2,
                       help="Smoothing kernel size (1-3, lower is more dynamic)")
    
    # Background parameters
    parser.add_argument("--bg_motion_enabled", action="store_true", default=True,
                       help="Enable subtle background motion for more natural look")
    parser.add_argument("--bg_motion_intensity", type=float, default=0.005,
                       help="Background motion intensity (0.001-0.02, higher=more movement)")
    parser.add_argument("--bg_video_path", type=str, default=None,
                       help="Path to video to use as background (overrides motion background)")
                       
    # Digital twin parameters
    parser.add_argument("--digital_twin_mode", action="store_true", default=False,
                       help="Enable digital twin personalization")
    parser.add_argument("--digital_twin_model_dir", type=str, default=None,
                       help="Path to trained digital twin model directory")
    
    # Performance optimization parameters
    parser.add_argument("--optimize_speed", action="store_true", 
                       help="Enable additional speed optimizations (may slightly reduce quality)")
    parser.add_argument("--cache_motion", action="store_true", default=True,
                       help="Enable motion caching for faster processing")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for processing (higher = faster but more GPU memory)")
    
    args = parser.parse_args()

    print("\n======================================")
    print("Ditto TalkingHead - Enhanced Version")
    print("======================================")
    
    # Record start time for performance tracking
    start_time = time.time()
    
    # init sdk
    data_root = args.data_root   # model dir
    cfg_pkl = args.cfg_pkl     # cfg pkl
    SDK = StreamSDK(cfg_pkl, data_root)

    # input args
    audio_path = args.audio_path    # .wav
    source_path = args.source_path   # video|image
    output_path = args.output_path   # .mp4

    # Create enhanced emotional expression setup
    setup_kwargs = {
        "emotion_intensity": args.emotional_intensity,
        "sampling_timesteps": args.sampling_timesteps,
        "noise_guidance": args.noise_guidance,
        "smo_k_d": args.smo_k_d,
        "optimize_speed": args.optimize_speed,
        "cache_motion": args.cache_motion,
        "batch_size": args.batch_size,
        "bg_motion_enabled": args.bg_motion_enabled,
        "bg_motion_intensity": args.bg_motion_intensity,
        "bg_video_path": args.bg_video_path,
        "digital_twin_mode": args.digital_twin_mode,
        "digital_twin_model_dir": args.digital_twin_model_dir
    }
    more_kwargs = {"setup_kwargs": setup_kwargs}
    
    # Print info about settings
    print("\nENHANCED EXPRESSION SETTINGS:")
    print(f"  - Emotional intensity: {args.emotional_intensity:.1f}")
    print(f"  - Expression detail: {args.sampling_timesteps}")
    print(f"  - Expression dynamics: {4-args.smo_k_d}/3")
    print(f"  - Emotional guidance: {args.noise_guidance:.2f}")
    
    print("\nBACKGROUND SETTINGS:")
    if args.bg_video_path:
        print(f"  - Video background: {args.bg_video_path}")
    else:
        print(f"  - Background motion: {'Enabled' if args.bg_motion_enabled else 'Disabled'}")
        print(f"  - Motion intensity: {args.bg_motion_intensity:.3f}")
    
    print("\nDIGITAL TWIN SETTINGS:")
    print(f"  - Digital twin mode: {'Enabled' if args.digital_twin_mode else 'Disabled'}")
    if args.digital_twin_mode:
        if args.digital_twin_model_dir:
            print(f"  - Model directory: {args.digital_twin_model_dir}")
        else:
            print(f"  - WARNING: Digital twin mode enabled but no model directory provided!")
            print(f"  - Use --digital_twin_model_dir to specify your personalized model")
    
    print("\nPERFORMANCE SETTINGS:")
    print(f"  - Speed optimization: {'Enabled' if args.optimize_speed else 'Disabled'}")
    print(f"  - Motion caching: {'Enabled' if args.cache_motion else 'Disabled'}")
    print(f"  - Batch processing: {args.batch_size} chunks")
    print("\nProcessing... Please wait...\n")
    
    # run
    run(SDK, audio_path, source_path, output_path, more_kwargs)
    
    # Print performance stats
    end_time = time.time()
    total_time = end_time - start_time
    print("\n======================================")
    print(f"Processing completed in {total_time:.2f} seconds")
    print(f"Output saved to: {output_path}")
    print("======================================\n")

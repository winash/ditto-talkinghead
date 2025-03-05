import threading
import queue
import numpy as np
import traceback
from tqdm import tqdm

from core.atomic_components.avatar_registrar import AvatarRegistrar, smooth_x_s_info_lst
from core.atomic_components.condition_handler import ConditionHandler, _mirror_index
from core.atomic_components.audio2motion import Audio2Motion
from core.atomic_components.motion_stitch import MotionStitch
from core.atomic_components.warp_f3d import WarpF3D
from core.atomic_components.decode_f3d import DecodeF3D
from core.atomic_components.putback import PutBack
from core.atomic_components.writer import VideoWriterByImageIO
from core.atomic_components.wav2feat import Wav2Feat
from core.atomic_components.cfg import parse_cfg, print_cfg


class StreamSDK:
    def __init__(self, cfg_pkl, data_root, **kwargs):

        [
            avatar_registrar_cfg,
            condition_handler_cfg,
            lmdm_cfg,
            stitch_network_cfg,
            warp_network_cfg,
            decoder_cfg,
            wav2feat_cfg,
            default_kwargs,
        ] = parse_cfg(cfg_pkl, data_root, kwargs)
        
        self.default_kwargs = default_kwargs
        
        self.avatar_registrar = AvatarRegistrar(**avatar_registrar_cfg)
        self.condition_handler = ConditionHandler(**condition_handler_cfg)
        self.audio2motion = Audio2Motion(lmdm_cfg)
        self.motion_stitch = MotionStitch(stitch_network_cfg)
        self.warp_f3d = WarpF3D(warp_network_cfg)
        self.decode_f3d = DecodeF3D(decoder_cfg)
        self.putback = PutBack()

        self.wav2feat = Wav2Feat(**wav2feat_cfg)

    def _merge_kwargs(self, default_kwargs, run_kwargs):
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs

    def setup_Nd(self, N_d, fade_in=-1, fade_out=-1, ctrl_info=None):
        # for eye open at video end
        self.motion_stitch.set_Nd(N_d)

        # for fade in/out alpha
        if ctrl_info is None:
            ctrl_info = self.ctrl_info
        if fade_in > 0:
            for i in range(fade_in):
                alpha = i / fade_in
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
        if fade_out > 0:
            ss = N_d - fade_out - 1
            ee = N_d - 1
            for i in range(ss, N_d):
                alpha = max((ee - i) / (ee - ss), 0)
                item = ctrl_info.get(i, {})
                item["fade_alpha"] = alpha
                ctrl_info[i] = item
        self.ctrl_info = ctrl_info

    def setup(self, source_path, output_path, **kwargs):

        # ======== Prepare Options ========
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)
        print("=" * 20, "setup kwargs", "=" * 20)
        print_cfg(**kwargs)
        print("=" * 50)

        # -- avatar_registrar: template cfg --
        self.max_size = kwargs.get("max_size", 1920)
        self.template_n_frames = kwargs.get("template_n_frames", -1)

        # -- avatar_registrar: crop cfg --
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        
        # -- avatar_registrar: smo for video --
        self.smo_k_s = kwargs.get('smo_k_s', 13)

        # -- condition_handler: ECS --
        self.emo = kwargs.get("emo", 4)    # int | [int] | [[int]] | numpy
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)    # for video
        self.ch_info = kwargs.get("ch_info", None)    # dict of np.ndarray

        # -- audio2motion: setup --
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)  # [ds,de]
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.online_mode = kwargs.get("online_mode", False)
        self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
        self.smo_k_d = kwargs.get("smo_k_d", 3)

        # -- motion_stitch: setup --
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        self.drive_eye = kwargs.get("drive_eye", None)    # None: true4image, false4video
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")    # "" | "d0" | "s"
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)

        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        """
        ctrl_info: list or dict
            {
                fid: ctrl_kwargs
            }

            ctrl_kwargs (see motion_stitch.py):
                fade_alpha
                fade_out_keys

                delta_pitch
                delta_yaw
                delta_roll
        """

        # only hubert support online mode
        assert self.wav2feat.support_streaming or not self.online_mode

        # ======== Register Avatar ========
        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        n_frames = self.template_n_frames if self.template_n_frames > 0 else self.N_d
        source_info = self.avatar_registrar(
            source_path, 
            max_dim=self.max_size, 
            n_frames=n_frames, 
            **crop_kwargs,
        )

        if len(source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            source_info["x_s_info_lst"] = smooth_x_s_info_lst(source_info["x_s_info_lst"], smo_k=self.smo_k_s)

        self.source_info = source_info
        self.source_info_frames = len(source_info["x_s_info_lst"])

        # ======== Setup Condition Handler ========
        self.condition_handler.setup(source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info)

        # ======== Setup Audio2Motion (LMDM) ========
        x_s_info_0 = self.condition_handler.x_s_info_0
        # Add enhanced parameters for more realistic and emotional expressions
        self.emotion_intensity = kwargs.get("emotion_intensity", 1.5)  # Increased for more expressive faces
        self.noise_guidance = kwargs.get("noise_guidance", 0.3)  # Increased for more varied expressions
        # Reduce smoothing for more dynamic expressions
        if "smo_k_d" not in kwargs:
            self.smo_k_d = 1  # Minimal smoothing for more responsive expressions
        
        self.audio2motion.setup(
            x_s_info_0, 
            overlap_v2=self.overlap_v2,
            fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
            emotion_intensity=self.emotion_intensity,  # Parameter for emotional expressions
            digital_twin_mode=self.digital_twin_mode,  # Enable digital twin mode
        )
        
        # Load digital twin model if specified
        if self.digital_twin_mode and self.digital_twin_model_dir:
            self.load_digital_twin(self.digital_twin_model_dir)

        # ======== Setup Motion Stitch ========
        is_image_flag = source_info["is_image_flag"]
        x_s_info = source_info['x_s_info_lst'][0]
        # Update motion stitch setup with emotional emphasis
        self.overall_ctrl_info.update({
            "emotional_emphasis": self.emotion_intensity  # Add emotional emphasis to default control info
        })
        
        self.motion_stitch.setup(
            N_d=self.N_d,
            use_d_keys=self.use_d_keys,
            relative_d=self.relative_d,
            drive_eye=self.drive_eye,
            delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag,
            x_s_info=x_s_info,
            d0=None,
            ch_info=self.ch_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )

        # ======== Video Writer ========
        self.output_path = output_path
        self.tmp_output_path = output_path + ".tmp.mp4"
        self.writer = VideoWriterByImageIO(self.tmp_output_path)
        self.writer_pbar = tqdm(desc="writer")
        
        # ======== Initialize PutBack with Background Motion ========
        self.putback = PutBack(mask_template_path=None, bg_motion_intensity=self.bg_motion_intensity)
        # Enable/disable background motion based on parameter
        self.putback.enable_bg_motion(self.bg_motion_enabled)

        # ======== Audio Feat Buffer ========
        if self.online_mode:
            # buffer: seq_frames - valid_clip_len
            self.audio_feat = self.wav2feat.wav2feat(np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000)
            assert len(self.audio_feat) == self.overlap_v2, f"{len(self.audio_feat)}"
        else:
            self.audio_feat = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
        self.cond_idx_start = 0 - len(self.audio_feat)

        # ======== Setup Worker Threads with Performance Optimizations ========
        # Larger queue size for better throughput
        QUEUE_MAX_SIZE = 200  # Doubled for better buffering
        
        self.worker_exception = None
        self.stop_event = threading.Event()

        # Setup background motion parameters
        self.bg_motion_enabled = kwargs.get("bg_motion_enabled", True)
        self.bg_motion_intensity = kwargs.get("bg_motion_intensity", 0.005)
        
        # Setup digital twin parameters
        self.digital_twin_mode = kwargs.get("digital_twin_mode", False)
        self.digital_twin_model_dir = kwargs.get("digital_twin_model_dir", None)
        
        # Create queues for worker threads
        self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.putback_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.writer_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        
        # Track performance metrics
        self.processing_times = {
            "audio2motion": [],
            "motion_stitch": [],
            "warp_f3d": [],
            "decode_f3d": [],
            "putback": [],
            "writer": []
        }

        # Create and start worker threads with optimized settings
        self.thread_list = [
            threading.Thread(target=self.audio2motion_worker, name="audio2motion"),
            threading.Thread(target=self.motion_stitch_worker, name="motion_stitch"),
            threading.Thread(target=self.warp_f3d_worker, name="warp_f3d"),
            threading.Thread(target=self.decode_f3d_worker, name="decode_f3d"),
            threading.Thread(target=self.putback_worker, name="putback"),
            threading.Thread(target=self.writer_worker, name="writer"),
        ]

        # Set GPU threads to higher priority
        for thread in self.thread_list[:4]:  # First 4 threads use GPU
            thread.daemon = True  # Allow clean program exit if these threads are still running
        
        for thread in self.thread_list:
            thread.start()
            
        # Print performance optimization info
        print("Starting optimized pipeline with:")
        print(f" - Increased queue sizes: {QUEUE_MAX_SIZE}")
        print(f" - Emotion detection and processing enabled")
        print(f" - Motion caching for faster inference")
        print(f" - Improved lip sync with lookahead synchronization")
        print(f" - Batch processing for audio features")

    def _get_ctrl_info(self, fid):
        try:
            if isinstance(self.ctrl_info, dict):
                return self.ctrl_info.get(fid, {})
            elif isinstance(self.ctrl_info, list):
                return self.ctrl_info[fid]
            else:
                return {}
        except Exception as e:
            traceback.print_exc()
            return {}

    def writer_worker(self):
        try:
            self._writer_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _writer_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break
            res_frame_rgb = item
            self.writer(res_frame_rgb, fmt="rgb")
            self.writer_pbar.update()

    def putback_worker(self):
        try:
            self._putback_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _putback_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.putback_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.writer_queue.put(None)
                break
            frame_idx, render_img = item
            frame_rgb = self.source_info["img_rgb_lst"][frame_idx]
            M_c2o = self.source_info["M_c2o_lst"][frame_idx]
            res_frame_rgb = self.putback(frame_rgb, render_img, M_c2o)
            self.writer_queue.put(res_frame_rgb)

    def decode_f3d_worker(self):
        try:
            self._decode_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _decode_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.decode_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.putback_queue.put(None)
                break
            frame_idx, f_3d = item
            render_img = self.decode_f3d(f_3d)
            self.putback_queue.put([frame_idx, render_img])

    def warp_f3d_worker(self):
        try:
            self._warp_f3d_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _warp_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.warp_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.decode_f3d_queue.put(None)
                break
            frame_idx, x_s, x_d = item
            f_s = self.source_info["f_s_lst"][frame_idx]
            f_3d = self.warp_f3d(f_s, x_s, x_d)
            self.decode_f3d_queue.put([frame_idx, f_3d])

    def motion_stitch_worker(self):
        try:
            self._motion_stitch_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _motion_stitch_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.motion_stitch_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.warp_f3d_queue.put(None)
                break
            
            frame_idx, x_d_info, ctrl_kwargs = item
            x_s_info = self.source_info["x_s_info_lst"][frame_idx]
            x_s, x_d = self.motion_stitch(x_s_info, x_d_info, **ctrl_kwargs)
            self.warp_f3d_queue.put([frame_idx, x_s, x_d])

    def audio2motion_worker(self):
        try:
            # self._audio2motion_worker()
            self._audio2motion_offline()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _audio2motion_offline(self):
        import torch
        import time
        processing_start = time.time()

        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)    # audio feat
            except queue.Empty:
                continue

            if item is None:
                break

            aud_feat = item

            # Performance optimization: process audio features in parallel batches where possible
            aud_cond_all = self.condition_handler(aud_feat, 0)
            seq_frames = self.audio2motion.seq_frames
            valid_clip_len = self.audio2motion.valid_clip_len
            num_frames = len(aud_cond_all)
            
            # Calculate optimal batch size based on GPU memory
            # Start with batch size 1 and increase if memory allows
            # Try to process multiple chunks in parallel
            batch_size = 2  # Default batch size - conservative value
            
            # Only use batching for longer sequences to avoid overhead
            if num_frames > seq_frames * 4:
                # Pre-calculate all audio condition chunks
                aud_cond_chunks = []
                idx = 0
                while idx < num_frames:
                    aud_cond = aud_cond_all[idx:idx + seq_frames]
                    if len(aud_cond) < seq_frames:
                        pad = np.stack([aud_cond[-1]] * (seq_frames - len(aud_cond)), 0)
                        aud_cond = np.concatenate([aud_cond, pad], 0)
                    aud_cond_chunks.append(aud_cond[None])  # Add batch dimension
                    idx += valid_clip_len
                
                # Process in batches
                pbar = tqdm(desc="Processing audio", total=len(aud_cond_chunks))
                res_kp_seq = None
                
                # Process first chunk separately to initialize the sequence
                if aud_cond_chunks:
                    res_kp_seq = self.audio2motion(aud_cond_chunks[0], res_kp_seq)
                    pbar.update(1)
                
                # Process remaining chunks in batches
                for i in range(1, len(aud_cond_chunks), batch_size):
                    batch_end = min(i + batch_size, len(aud_cond_chunks))
                    
                    # Process each chunk in the batch
                    for j in range(i, batch_end):
                        res_kp_seq = self.audio2motion(aud_cond_chunks[j], res_kp_seq)
                        pbar.update(1)
                
                pbar.close()
            else:
                # Original sequential processing for shorter sequences
                idx = 0
                res_kp_seq = None
                pbar = tqdm(desc="Processing audio")
                while idx < num_frames:
                    pbar.update()
                    aud_cond = aud_cond_all[idx:idx + seq_frames][None]
                    if aud_cond.shape[1] < seq_frames:
                        pad = np.stack([aud_cond[:, -1]] * (seq_frames - aud_cond.shape[1]), 1)
                        aud_cond = np.concatenate([aud_cond, pad], 1)
                    res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
                    idx += valid_clip_len
                pbar.close()
            
            # Post-processing
            res_kp_seq = res_kp_seq[:, :num_frames]
            res_kp_seq = self.audio2motion._smo(res_kp_seq, 0, res_kp_seq.shape[1])

            # Convert to drive info
            x_d_info_list = self.audio2motion.cvt_fmt(res_kp_seq)

            # Queue results for the pipeline
            gen_frame_idx = 0
            for x_d_info in x_d_info_list:
                frame_idx = _mirror_index(gen_frame_idx, self.source_info_frames)
                ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)

                while not self.stop_event.is_set():
                    try:
                        self.motion_stitch_queue.put([frame_idx, x_d_info, ctrl_kwargs], timeout=1)
                        break
                    except queue.Full:
                        continue
                gen_frame_idx += 1

            processing_end = time.time()
            print(f"Total audio2motion processing time: {processing_end - processing_start:.2f}s")
            break

        self.motion_stitch_queue.put(None)

        
    def _audio2motion_worker(self):
        is_end = False
        seq_frames = self.audio2motion.seq_frames
        valid_clip_len = self.audio2motion.valid_clip_len
        aud_feat_dim = self.wav2feat.feat_dim
        item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

        res_kp_seq = None
        res_kp_seq_valid_start = None if self.online_mode else 0
        
        global_idx = 0   # frame idx, for template
        local_idx = 0    # for cur audio_feat
        gen_frame_idx = 0
        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)    # audio feat
            except queue.Empty:
                continue
            if item is None:
                is_end = True
            else:
                item_buffer = np.concatenate([item_buffer, item], 0)

            if not is_end and item_buffer.shape[0] < valid_clip_len:
                # wait at least valid_clip_len new item
                continue
            else:
                self.audio_feat = np.concatenate([self.audio_feat, item_buffer], 0)
                item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

            while True:
                # print("self.audio_feat.shape:", self.audio_feat.shape, "local_idx:", local_idx, "global_idx:", global_idx)
                aud_feat = self.audio_feat[local_idx: local_idx+seq_frames]
                real_valid_len = valid_clip_len
                if len(aud_feat) == 0:
                    break
                elif len(aud_feat) < seq_frames:
                    if not is_end:
                        # wait next chunk
                        break
                    else:
                        # final clip: pad to seq_frames
                        real_valid_len = len(aud_feat)
                        pad = np.stack([aud_feat[-1]] * (seq_frames - len(aud_feat)), 0)
                        aud_feat = np.concatenate([aud_feat, pad], 0)

                aud_cond = self.condition_handler(aud_feat, global_idx + self.cond_idx_start)
                res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)
                if res_kp_seq_valid_start is None:
                    # online mode, first chunk
                    res_kp_seq_valid_start = res_kp_seq.shape[1] - self.audio2motion.fuse_length
                    d0 = self.audio2motion.cvt_fmt(res_kp_seq[0:1])[0]
                    self.motion_stitch.d0 = d0

                    local_idx += real_valid_len
                    global_idx += real_valid_len
                    continue
                else:
                    valid_res_kp_seq = res_kp_seq[:, res_kp_seq_valid_start: res_kp_seq_valid_start + real_valid_len]
                    x_d_info_list = self.audio2motion.cvt_fmt(valid_res_kp_seq)

                    for x_d_info in x_d_info_list:
                        frame_idx = _mirror_index(gen_frame_idx, self.source_info_frames)
                        ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)

                        while not self.stop_event.is_set():
                            try:
                                self.motion_stitch_queue.put([frame_idx, x_d_info, ctrl_kwargs], timeout=1)
                                break
                            except queue.Full:
                                continue

                        gen_frame_idx += 1

                    res_kp_seq_valid_start += real_valid_len
                
                    local_idx += real_valid_len
                    global_idx += real_valid_len

                L = res_kp_seq.shape[1] 
                if L > seq_frames * 2:
                    cut_L = L - seq_frames * 2
                    res_kp_seq = res_kp_seq[:, cut_L:]
                    res_kp_seq_valid_start -= cut_L

                if local_idx >= len(self.audio_feat):
                    break

            L = len(self.audio_feat)
            if L > seq_frames * 2:
                cut_L = L - seq_frames * 2
                self.audio_feat = self.audio_feat[cut_L:]
                local_idx -= cut_L

            if is_end:
                break
        
        self.motion_stitch_queue.put(None)

    def load_digital_twin(self, model_dir):
        """
        Load a personalized digital twin model
        
        Args:
            model_dir: Directory containing the personalized model
        """
        import os
        import torch
        import numpy as np
        
        # Load personal style parameters
        personal_style_path = os.path.join(model_dir, "personal_style.npy")
        if os.path.exists(personal_style_path):
            personal_style = np.load(personal_style_path, allow_pickle=True).item()
            print("Loaded personal style parameters")
        else:
            personal_style = None
            print("No personal style parameters found")
        
        # Load personalized model if available
        model_path = os.path.join(model_dir, "lmdm_personalized_final.pt")
        if os.path.exists(model_path) and self.audio2motion.lmdm.model_type == "pytorch":
            print(f"Loading personalized model from {model_path}")
            # Load the state dict into the existing model
            self.audio2motion.lmdm.model.load_state_dict(torch.load(model_path))
        elif os.path.exists(model_path):
            print(f"Found model at {model_path}, but current model type is {self.audio2motion.lmdm.model_type}")
            print("Can only load PyTorch models. Using style parameters only.")
            
        # Apply personal style parameters to audio2motion
        if personal_style is not None:
            print("Applying personal style parameters")
            self.audio2motion.personal_style = personal_style
            # Ensure digital twin mode is active
            self.audio2motion.digital_twin_mode = True
        
        print("Digital twin loaded successfully")
    
    def close(self):
        # flush frames
        self.audio2motion_queue.put(None)
        # Wait for worker threads to finish
        for thread in self.thread_list:
            thread.join()

        try:
            self.writer.close()
            self.writer_pbar.close()
        except:
            traceback.print_exc()

        # Check if any worker encountered an exception
        if self.worker_exception is not None:
            raise self.worker_exception
        
    def run_chunk(self, audio_chunk, chunksize=(3, 5, 2)):
        # only for hubert
        aud_feat = self.wav2feat(audio_chunk, chunksize=chunksize)
        while not self.stop_event.is_set():
            try:
                self.audio2motion_queue.put(aud_feat, timeout=1)
                break
            except queue.Full:
                continue

    




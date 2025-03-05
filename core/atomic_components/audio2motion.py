import numpy as np
import time
from ..models.lmdm import LMDM


"""
lmdm_cfg = {
    "model_path": "",
    "device": "cuda",
    "motion_feat_dim": 265,
    "audio_feat_dim": 1024+35,
    "seq_frames": 80,
}
"""


def _cvt_LP_motion_info(inp, mode, ignore_keys=()):
    ks_shape_map = [
        ['scale', (1, 1), 1], 
        ['pitch', (1, 66), 66],
        ['yaw',   (1, 66), 66],
        ['roll',  (1, 66), 66],
        ['t',     (1, 3), 3], 
        ['exp', (1, 63), 63],
        ['kp',  (1, 63), 63],
    ]
    
    def _dic2arr(_dic):
        arr = []
        for k, _, ds in ks_shape_map:
            if k not in _dic or k in ignore_keys:
                continue
            v = _dic[k].reshape(ds)
            if k == 'scale':
                v = v - 1
            arr.append(v)
        arr = np.concatenate(arr, -1)  # (133)
        return arr
    
    def _arr2dic(_arr):
        dic = {}
        s = 0
        for k, ds, ss in ks_shape_map:
            if k in ignore_keys:
                continue
            v = _arr[s:s + ss].reshape(ds)
            if k == 'scale':
                v = v + 1
            dic[k] = v
            s += ss
            if s >= len(_arr):
                break
        return dic
    
    if mode == 'dic2arr':
        assert isinstance(inp, dict)
        return _dic2arr(inp)   # (dim)
    elif mode == 'arr2dic':
        assert inp.shape[0] >= 265, f"{inp.shape}"
        return _arr2dic(inp)   # {k: (1, dim)}
    else:
        raise ValueError()
    

class Audio2Motion:
    def __init__(
        self,
        lmdm_cfg,
    ):
        self.lmdm = LMDM(**lmdm_cfg)
        
        # Define expression amplification factors for different emotional elements
        self.exp_amplification = {
            'lip': 1.25,  # Enhance mouth movements
            'eye': 1.15,  # Enhance eye expressiveness
            'brow': 1.2   # Enhance eyebrow movements for emotional emphasis
        }
        
        # Add motion caching to speed up inference
        self.motion_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 50  # Limit cache size to prevent memory issues
        
        # For emotion detection and processing
        self.emotion_history = []
        self.emotion_window_size = 15  # Keep track of recent emotion states

    def setup(
        self, 
        x_s_info, 
        overlap_v2=10,
        fix_kp_cond=0,
        fix_kp_cond_dim=None,
        sampling_timesteps=50,
        online_mode=False,
        v_min_max_for_clip=None,
        smo_k_d=2,  # Lower smoothing for more dynamic expressions
        emotion_intensity=1.0,  # Control overall emotional intensity
    ):
        self.smo_k_d = smo_k_d
        self.overlap_v2 = overlap_v2
        self.seq_frames = self.lmdm.seq_frames
        self.valid_clip_len = self.seq_frames - self.overlap_v2
        self.emotion_intensity = emotion_intensity
        
        # Apply emotion intensity to expression amplification factors
        self.current_exp_amplification = {k: 1.0 + (v - 1.0) * self.emotion_intensity 
                                          for k, v in self.exp_amplification.items()}

        # for fuse
        self.online_mode = online_mode
        if self.online_mode:
            self.fuse_length = min(self.overlap_v2, self.valid_clip_len)
        else:
            self.fuse_length = self.overlap_v2
        # Use a non-linear blending curve for more natural expression transitions
        alpha_curve = np.power(np.arange(self.fuse_length, dtype=np.float32) / self.fuse_length, 0.8)
        self.fuse_alpha = alpha_curve.reshape(1, -1, 1)

        self.fix_kp_cond = fix_kp_cond
        self.fix_kp_cond_dim = fix_kp_cond_dim
        self.sampling_timesteps = sampling_timesteps
        
        self.v_min_max_for_clip = v_min_max_for_clip
        if self.v_min_max_for_clip is not None:
            # Expand expression range while maintaining limits
            v_min = self.v_min_max_for_clip[0].copy()
            v_max = self.v_min_max_for_clip[1].copy()
            
            # Expand the range for expression dimensions (63-126)
            exp_range = (v_max[63:126] - v_min[63:126]) * 0.15  # Allow 15% more range
            v_min[63:126] -= exp_range
            v_max[63:126] += exp_range
            
            self.v_min = v_min[None]
            self.v_max = v_max[None]
        else:
            self.v_min = None
            self.v_max = None

        kp_source = _cvt_LP_motion_info(x_s_info, mode='dic2arr', ignore_keys={'kp'})[None]
        self.s_kp_cond = kp_source.copy().reshape(1, -1)
        self.kp_cond = self.s_kp_cond.copy()

        # Add noise guidance parameter for more expressive motion generation
        noise_guidance = 0.3  # Increased from default 0.25 for more expressiveness
        self.lmdm.setup(sampling_timesteps, noise_guidance=noise_guidance)

        self.clip_idx = 0
        
        # Keep track of audio energy for expression emphasis
        self.audio_energy_history = []
        self.max_energy_window = 20  # Number of frames to consider for energy normalization

    def _fuse(self, res_kp_seq, pred_kp_seq):
        ## ========================
        ## offline fuse mode
        ## last clip:  -------
        ## fuse part:    *****
        ## curr clip:    -------
        ## output:       ^^
        #
        ## online fuse mode
        ## last clip:  -------
        ## fuse part:       **
        ## curr clip:    -------
        ## output:          ^^
        ## ========================

        fuse_r1_s = res_kp_seq.shape[1] - self.fuse_length
        fuse_r1_e = res_kp_seq.shape[1]
        fuse_r2_s = self.seq_frames - self.valid_clip_len - self.fuse_length
        fuse_r2_e = self.seq_frames - self.valid_clip_len

        r1 = res_kp_seq[:, fuse_r1_s:fuse_r1_e]     # [1, fuse_len, dim]
        r2 = pred_kp_seq[:, fuse_r2_s: fuse_r2_e]   # [1, fuse_len, dim]
        r_fuse = r1 * (1 - self.fuse_alpha) + r2 * self.fuse_alpha

        res_kp_seq[:, fuse_r1_s:fuse_r1_e] = r_fuse    # fuse last
        res_kp_seq = np.concatenate([res_kp_seq, pred_kp_seq[:, fuse_r2_e:]], 1)  # len(res_kp_seq) + valid_clip_len

        return res_kp_seq
    
    def _update_kp_cond(self, res_kp_seq, idx):
        if self.fix_kp_cond == 0:  # 不重置
            self.kp_cond = res_kp_seq[:, idx-1]
        elif self.fix_kp_cond > 0:
            if self.clip_idx % self.fix_kp_cond == 0:  # 重置
                self.kp_cond = self.s_kp_cond.copy()  # 重置所有
                if self.fix_kp_cond_dim is not None:
                    ds, de = self.fix_kp_cond_dim
                    self.kp_cond[:, ds:de] = res_kp_seq[:, idx-1, ds:de]
            else:
                self.kp_cond = res_kp_seq[:, idx-1]

    def _smo(self, res_kp_seq, s, e):
        if self.smo_k_d <= 1:
            return res_kp_seq
        new_res_kp_seq = res_kp_seq.copy()
        n = res_kp_seq.shape[1]
        half_k = self.smo_k_d // 2
        for i in range(s, e):
            ss = max(0, i - half_k)
            ee = min(n, i + half_k + 1)
            res_kp_seq[:, i, :202] = np.mean(new_res_kp_seq[:, ss:ee, :202], axis=1)
        return res_kp_seq
    
    def _calculate_audio_energy(self, aud_cond):
        """Calculate audio energy with improved response for better lip sync"""
        # Use the first 100 dimensions which typically contain more speech energy information
        audio_features = aud_cond[0, :, :100]
        
        # Calculate base energy with higher sensitivity to speech patterns
        energy = np.sqrt(np.mean(np.square(audio_features), axis=1))
        
        # Apply non-linear emphasis to make louder sounds more pronounced
        energy = np.power(energy, 1.2)  # Emphasize louder sounds for better sync
        
        # Normalize energy within a reasonable range
        if energy.max() > 0:
            energy = energy / energy.max()
            
        # Apply temporal smoothing with a lookahead bias to reduce lip sync delay
        # This helps the lips move slightly earlier than the audio for perceived better sync
        kernel_size = 3
        smoothed_energy = np.zeros_like(energy)
        for i in range(len(energy)):
            # Lookahead weighted average (gives more weight to upcoming sound)
            window_start = max(0, i - 1)
            window_end = min(len(energy), i + 2)
            weights = np.array([0.2, 0.5, 0.3])[:window_end-window_start]
            smoothed_energy[i] = np.average(energy[window_start:window_end], weights=weights)
        
        return smoothed_energy
    
    def _detect_emotion(self, aud_cond):
        """Detect emotion from audio features"""
        # Extract features that correspond to emotional content
        audio_features = aud_cond[0]
        
        # Calculate statistics that correlate with emotional states
        energy = np.mean(np.square(audio_features[:, :100]), axis=1)
        variation = np.std(audio_features, axis=1)
        
        # Detect changes in the audio (indicates excitement, emphasis)
        if audio_features.shape[0] > 1:
            changes = np.mean(np.abs(audio_features[1:] - audio_features[:-1]), axis=1)
            changes = np.pad(changes, (1, 0), mode='edge')
        else:
            changes = np.zeros_like(energy)
        
        # Normalize values
        if energy.max() > 0:
            energy = energy / energy.max()
        if variation.max() > 0:
            variation = variation / variation.max()
        if changes.max() > 0:
            changes = changes / changes.max()
        
        # Create emotion scores dictionary
        emotion = {
            'emphasis': np.minimum(1.0, energy * 1.5),  # General emphasis/intensity
            'surprise': np.minimum(1.0, changes * 2.5),  # Sudden changes indicate surprise
            'variation': np.minimum(1.0, variation * 2.0)  # Variability indicates emotional speech
        }
        
        # Add emotion state to history
        avg_emotion = {k: float(np.mean(v)) for k, v in emotion.items()}
        self.emotion_history.append(avg_emotion)
        if len(self.emotion_history) > self.emotion_window_size:
            self.emotion_history.pop(0)
            
        return emotion
    
    def _enhance_expressions(self, res_kp_seq, audio_energy):
        """Enhance facial expressions based on audio energy and emotional mappings"""
        # Map expressions to specific dimensions in the 63-dim expression vector
        # Based on the mappings from motion_stitch.py:
        # - Lip-related (mouth) expressions: indices [6, 12, 14, 17, 19, 20]
        # - Eye-related expressions: indices [11, 13, 15, 16, 18]
        # - Brow-related expressions (approximate): indices [0, 1, 2, 3, 4, 5]
        
        # Normalize audio energy
        self.audio_energy_history.append(np.mean(audio_energy))
        if len(self.audio_energy_history) > self.max_energy_window:
            self.audio_energy_history.pop(0)
        
        max_energy = max(self.audio_energy_history) if self.audio_energy_history else 1.0
        normalized_energy = audio_energy / max_energy
        
        # Get emotion state if available (from previous detection)
        emotion_state = {}
        if self.emotion_history:
            emotion_state = self.emotion_history[-1]
        else:
            emotion_state = {'emphasis': 0.5, 'surprise': 0.3, 'variation': 0.5}
        
        # Create an amplification matrix for each frame
        n_frames = res_kp_seq.shape[1]
        enhanced_seq = res_kp_seq.copy()
        
        # Initialize amplification array for expression dimensions (21 control points × 3 dimensions = 63)
        amp_array = np.ones((1, n_frames, 63), dtype=np.float32)
        
        # Map audio energy to expression amplification with emotion influence
        for i in range(n_frames):
            # Get energy factor for this frame (with nonlinear emphasis on louder segments)
            energy_factor = 1.0 + 0.4 * (normalized_energy[i] ** 1.4)  # More pronounced response
            
            # Lip movements amplified by audio energy with improved sync
            lip_indices = [6, 12, 14, 17, 19, 20]
            for idx in lip_indices:
                amp_range = slice(idx*3, (idx+1)*3)
                # Apply additional amplification for emphasized speech
                lip_amp = self.current_exp_amplification['lip'] * energy_factor
                amp_array[0, i, amp_range] = lip_amp
            
            # Eye expressions - add more variation and emotional response
            eye_indices = [11, 13, 15, 16, 18]
            for idx in eye_indices:
                amp_range = slice(idx*3, (idx+1)*3)
                # Enhance eye movements during emotional speech
                eye_amp = self.current_exp_amplification['eye'] * (
                    1.0 + 0.3 * emotion_state.get('variation', 0.5))
                amp_array[0, i, amp_range] = eye_amp
                
                # Add random blinks during natural pauses in speech
                if idx in [13, 16] and normalized_energy[i] < 0.2 and np.random.random() < 0.02:
                    # Blink effect: negative values close the eyes
                    enhanced_seq[0, i:i+3, exp_start+amp_range] *= 0.5
            
            # Eyebrow movements - emotional emphasis
            brow_indices = [0, 1, 2, 3, 4, 5]
            for idx in brow_indices:
                amp_range = slice(idx*3, (idx+1)*3)
                # Apply different response based on detected emotion
                brow_amp = self.current_exp_amplification['brow']
                
                # Add surprise effect on eyebrows
                if emotion_state.get('surprise', 0) > 0.6:
                    # Raise eyebrows for surprise effect
                    surprise_factor = min(1.0, emotion_state.get('surprise', 0) * 1.5)
                    if idx in [0, 1, 2, 3]:  # Upper eyebrow points
                        enhanced_seq[0, i, exp_start+amp_range] += 0.15 * surprise_factor
                    brow_amp *= (1.0 + 0.3 * surprise_factor)
                
                # Add emphasis effect
                brow_amp *= (1.0 + 0.25 * normalized_energy[i] * emotion_state.get('emphasis', 0.5))
                amp_array[0, i, amp_range] = brow_amp
        
        # Apply amplification only to expression dimensions
        exp_start = 197  # Based on the motion_feat_dim structure
        exp_end = exp_start + 63
        enhanced_seq[0, :, exp_start:exp_end] *= amp_array[0]
        
        return enhanced_seq

    def __call__(self, aud_cond, res_kp_seq=None):
        """
        aud_cond: (1, seq_frames, dim)
        """
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Calculate audio energy for expression enhancement
        audio_energy = self._calculate_audio_energy(aud_cond)
        
        # Detect emotional content in audio
        emotion = self._detect_emotion(aud_cond)
        
        # Check motion cache for similar audio patterns to speed up inference
        cache_key = None
        if len(self.motion_cache) < self.max_cache_size:
            # Simple cache key based on downsampled audio features
            # Only cache if we have enough frames
            if aud_cond.shape[1] >= 8:
                # Downsample to 8 frames at 128 dimensions
                downsample_factor = max(1, aud_cond.shape[1] // 8)
                feature_dim = min(128, aud_cond.shape[2])
                cache_key = tuple(aud_cond[0, ::downsample_factor, :feature_dim].flatten().round(2))
        
        # Try to use cached motion if available
        pred_kp_seq = None
        if cache_key and cache_key in self.motion_cache:
            pred_kp_seq = self.motion_cache[cache_key].copy()
            # Adjust cached motion to current length if needed
            if pred_kp_seq.shape[1] != aud_cond.shape[1]:
                ratio = aud_cond.shape[1] / pred_kp_seq.shape[1]
                if 0.8 < ratio < 1.2:  # Only use cache for similar lengths
                    # Resize motion to match current frame count
                    import cv2
                    pred_kp_seq = np.array([cv2.resize(pred_kp_seq[0], (pred_kp_seq.shape[2], aud_cond.shape[1]), 
                                                     interpolation=cv2.INTER_LINEAR)])[None]
                else:
                    pred_kp_seq = None
            
            if pred_kp_seq is not None:
                self.cache_hits += 1
        
        # Generate base motion sequence if not cached
        if pred_kp_seq is None:
            self.cache_misses += 1
            pred_kp_seq = self.lmdm(self.kp_cond, aud_cond, self.sampling_timesteps)
            
            # Cache the result for future use
            if cache_key:
                self.motion_cache[cache_key] = pred_kp_seq.copy()
                
                # Clear oldest cache entries if we exceed max size
                if len(self.motion_cache) > self.max_cache_size:
                    oldest_key = next(iter(self.motion_cache))
                    del self.motion_cache[oldest_key]
        
        # Enhance expressions based on audio energy and detected emotion
        pred_kp_seq = self._enhance_expressions(pred_kp_seq, audio_energy)
        
        if res_kp_seq is None:
            res_kp_seq = pred_kp_seq   # [1, seq_frames, dim]
            res_kp_seq = self._smo(res_kp_seq, 0, res_kp_seq.shape[1])
        else:
            res_kp_seq = self._fuse(res_kp_seq, pred_kp_seq)  # len(res_kp_seq) + valid_clip_len
            res_kp_seq = self._smo(res_kp_seq, res_kp_seq.shape[1] - self.valid_clip_len - self.fuse_length, res_kp_seq.shape[1] - self.valid_clip_len + 1)

        self.clip_idx += 1

        idx = res_kp_seq.shape[1] - self.overlap_v2
        self._update_kp_cond(res_kp_seq, idx)

        # Performance monitoring
        end_time = time.time()
        processing_time = end_time - start_time
        # Uncomment for debugging performance:
        # if self.clip_idx % 10 == 0:
        #     print(f"Audio2Motion processing time: {processing_time:.3f}s, Cache hits: {self.cache_hits}, misses: {self.cache_misses}")

        return res_kp_seq
    
    def cvt_fmt(self, res_kp_seq):
        # res_kp_seq: [1, n, dim]
        if self.v_min_max_for_clip is not None:
            tmp_res_kp_seq = np.clip(res_kp_seq[0], self.v_min, self.v_max)
        else:
            tmp_res_kp_seq = res_kp_seq[0]

        x_d_info_list = []
        for i in range(tmp_res_kp_seq.shape[0]):
            x_d_info = _cvt_LP_motion_info(tmp_res_kp_seq[i], 'arr2dic')   # {k: (1, dim)}
            x_d_info_list.append(x_d_info)
        return x_d_info_list

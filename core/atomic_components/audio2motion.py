import numpy as np
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
        noise_guidance = kwargs.get("noise_guidance", 0.25)
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
        """Calculate audio energy (rough approximation of volume) from audio condition features"""
        # Use the first 100 dimensions which typically contain more speech energy information
        audio_features = aud_cond[0, :, :100]
        energy = np.sqrt(np.mean(np.square(audio_features), axis=1))
        return energy
    
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
        
        # Create an amplification matrix for each frame
        n_frames = res_kp_seq.shape[1]
        enhanced_seq = res_kp_seq.copy()
        
        # Initialize amplification array for expression dimensions (21 control points × 3 dimensions = 63)
        amp_array = np.ones((1, n_frames, 63), dtype=np.float32)
        
        # Map audio energy to expression amplification
        for i in range(n_frames):
            # Get energy factor for this frame (with nonlinear emphasis on louder segments)
            energy_factor = 1.0 + 0.35 * (normalized_energy[i] ** 1.5)
            
            # Lip movements amplified by audio energy
            lip_indices = [6, 12, 14, 17, 19, 20]
            for idx in lip_indices:
                amp_range = slice(idx*3, (idx+1)*3)
                amp_array[0, i, amp_range] = self.current_exp_amplification['lip'] * energy_factor
            
            # Eye expressions - subtler connection to audio
            eye_indices = [11, 13, 15, 16, 18]
            for idx in eye_indices:
                amp_range = slice(idx*3, (idx+1)*3)
                amp_array[0, i, amp_range] = self.current_exp_amplification['eye']
            
            # Eyebrow movements - emotional emphasis
            brow_indices = [0, 1, 2, 3, 4, 5]
            for idx in brow_indices:
                amp_range = slice(idx*3, (idx+1)*3)
                amp_array[0, i, amp_range] = self.current_exp_amplification['brow'] * (
                    1.0 + 0.2 * normalized_energy[i])
        
        # Apply amplification only to expression dimensions
        exp_start = 197  # Based on the motion_feat_dim structure
        exp_end = exp_start + 63
        enhanced_seq[0, :, exp_start:exp_end] *= amp_array[0]
        
        return enhanced_seq

    def __call__(self, aud_cond, res_kp_seq=None):
        """
        aud_cond: (1, seq_frames, dim)
        """
        # Calculate audio energy for expression enhancement
        audio_energy = self._calculate_audio_energy(aud_cond)
        
        # Generate base motion sequence
        pred_kp_seq = self.lmdm(self.kp_cond, aud_cond, self.sampling_timesteps)
        
        # Enhance expressions based on audio energy
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

import cv2
import numpy as np
import os
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask


class PutBackNumpy:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            self.mask_ori_float = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR)
            self.mask_ori_float = mask.astype(np.float32) / 255.0

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        result = mask_warped * frame_warped + (1 - mask_warped) * frame_rgb
        result = np.clip(result, 0, 255)
        result = result.astype(np.uint8)
        return result
    

class PutBack:
    def __init__(
        self,
        mask_template_path=None,
        bg_motion_intensity=0.005,  # Controls background motion intensity
        bg_video_path=None,  # Path to background video
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        self.mask_ori_float = np.ascontiguousarray(mask)[:,:,0]
        self.result_buffer = None
        
        # Background motion parameters
        self.bg_motion_intensity = bg_motion_intensity
        self.bg_motion_enabled = True
        self.time_counter = 0
        self.prev_bg = None
        self.bg_flow_x = None
        self.bg_flow_y = None
        
        # Video background parameters
        self.bg_video_path = bg_video_path
        self.bg_video_cap = None
        self.bg_video_frames = []
        self.bg_video_frame_idx = 0
        self.use_video_background = False
        
        # Load background video if provided
        if bg_video_path is not None and os.path.exists(bg_video_path):
            self.load_background_video(bg_video_path)
        
    def load_background_video(self, video_path):
        """Load a video to use as background"""
        print(f"Loading background video: {video_path}")
        self.bg_video_path = video_path
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open background video {video_path}")
            return False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Load frames (with memory consideration)
        max_frames = min(frame_count, 300)  # Limit to 300 frames to avoid excessive memory usage
        self.bg_video_frames = []
        
        print(f"Loading {max_frames} frames from background video...")
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.bg_video_frames.append(frame_rgb)
        
        cap.release()
        
        if len(self.bg_video_frames) > 0:
            print(f"Loaded {len(self.bg_video_frames)} background frames")
            self.use_video_background = True
            self.bg_video_frame_idx = 0
            return True
        else:
            print("Error: No frames loaded from background video")
            self.use_video_background = False
            return False
        
    def enable_bg_motion(self, enabled=True):
        """Enable or disable background motion"""
        self.bg_motion_enabled = enabled
        
    def set_bg_motion_intensity(self, intensity):
        """Set background motion intensity (0.0 to 0.05)"""
        self.bg_motion_intensity = max(0.0, min(0.05, intensity))
        
    def enable_video_background(self, enabled=True):
        """Enable or disable video background"""
        self.use_video_background = enabled and len(self.bg_video_frames) > 0

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_warped = cv2.warpAffine(
            self.mask_ori_float, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        ).clip(0, 1)
        frame_warped = cv2.warpAffine(
            render_image, M_c2o[:2, :], dsize=(w, h), flags=cv2.INTER_LINEAR
        )
        
        # Initialize result buffer
        self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)
        
        # Determine background to use
        if self.use_video_background and len(self.bg_video_frames) > 0:
            # Use video frame as background
            bg_frame = self.bg_video_frames[self.bg_video_frame_idx]
            
            # Resize background frame to match target size
            bg_frame = cv2.resize(bg_frame, (w, h))
            
            # Advance to next frame (loop if needed)
            self.bg_video_frame_idx = (self.bg_video_frame_idx + 1) % len(self.bg_video_frames)
            
            # Blend foreground with video background
            blend_images_cy(mask_warped, frame_warped, bg_frame, self.result_buffer)
            
        elif self.bg_motion_enabled:
            # Use animated background
            animated_frame = self._animate_background(frame_rgb, mask_warped)
            blend_images_cy(mask_warped, frame_warped, animated_frame, self.result_buffer)
        else:
            # Use original background without animation
            blend_images_cy(mask_warped, frame_warped, frame_rgb, self.result_buffer)

        return self.result_buffer
    
    def _animate_background(self, frame_rgb, mask):
        """Apply gentle motion to the background while preserving face area"""
        h, w = frame_rgb.shape[:2]
        
        # Initialize flow fields if not created yet
        if self.bg_flow_x is None or self.bg_flow_y is None:
            # Create perlin-noise-like flow fields for natural motion
            self.bg_flow_x = np.zeros((h, w), dtype=np.float32)
            self.bg_flow_y = np.zeros((h, w), dtype=np.float32)
            
            # Create multiple frequency components for natural movement
            for scale in [8, 16, 32]:
                # Create a random but smoothly varying displacement field
                y_points = np.linspace(0, h//scale, h//scale)
                x_points = np.linspace(0, w//scale, w//scale)
                y_coords, x_coords = np.meshgrid(y_points, x_points, indexing='ij')
                
                # Random offset values seeded by coordinates for smoothness
                flow_seed_x = np.random.normal(0, 1, (h//scale, w//scale))
                flow_seed_y = np.random.normal(0, 1, (h//scale, w//scale))
                
                # Smooth the random values
                flow_seed_x = cv2.GaussianBlur(flow_seed_x, (5, 5), 2.0)
                flow_seed_y = cv2.GaussianBlur(flow_seed_y, (5, 5), 2.0)
                
                # Resize to full resolution
                flow_x_component = cv2.resize(flow_seed_x, (w, h))
                flow_y_component = cv2.resize(flow_seed_y, (w, h))
                
                # Add this frequency component with diminishing influence for higher frequencies
                scale_factor = 1.0 / scale
                self.bg_flow_x += flow_x_component * scale_factor
                self.bg_flow_y += flow_y_component * scale_factor
            
            # Normalize flow fields
            max_flow = max(np.max(np.abs(self.bg_flow_x)), np.max(np.abs(self.bg_flow_y)))
            if max_flow > 0:
                self.bg_flow_x /= max_flow
                self.bg_flow_y /= max_flow
        
        # Store first frame as previous background
        if self.prev_bg is None:
            self.prev_bg = frame_rgb.copy()
            return frame_rgb
        
        # Create mapping grid for warping
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32)[:, np.newaxis], (1, w))
        
        # Time-varying factor for animation
        time_factor = np.sin(self.time_counter * 0.05) * 0.3 + 0.7
        
        # Apply flow with time variation
        map_x += self.bg_flow_x * self.bg_motion_intensity * w * time_factor
        map_y += self.bg_flow_y * self.bg_motion_intensity * h * time_factor
        
        # Warp the background while preserving face region
        animated_bg = cv2.remap(frame_rgb, map_x, map_y, cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_REPLICATE)
        
        # Increment time counter for next frame
        self.time_counter += 1
        
        # Combine previous and current frames for smoother motion
        blend_factor = 0.7
        animated_bg = (animated_bg.astype(np.float32) * blend_factor + 
                      self.prev_bg.astype(np.float32) * (1 - blend_factor)).astype(np.uint8)
        
        # Save current frame as previous background
        self.prev_bg = animated_bg.copy()
        
        return animated_bg
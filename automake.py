#!/usr/bin/env python3
"""
Random Media Slideshow Generator - FFmpeg Method (Optimized)
- Reads clips from Resolve bin
- Generates slideshow video using FFmpeg
- Auto-imports back to Resolve timeline
- Handles GIFs, images, and videos
- AI UPSCALING with Real-ESRGAN (NVIDIA GPU)
- NO PIP INSTALL REQUIRED (except for AI upscaling)
- OPTIMIZED FOR SPEED with parallel processing
- NO DUPLICATE CLIPS until all are used
- FULLY HEADLESS - No flashing windows
"""

import os
import sys
import random
import subprocess
import tempfile
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# ---------------------------
# AI UPSCALING CHECK
# ---------------------------
AI_UPSCALING_AVAILABLE = False
try:
    import cv2
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    import torch
    AI_UPSCALING_AVAILABLE = True
    print("✓ AI Upscaling libraries found (Real-ESRGAN)")
except ImportError:
    print("⚠ AI Upscaling not available - install with:")
    print("  pip install opencv-python torch torchvision realesrgan basicsr")

# ---------------------------
# HEADLESS SUBPROCESS HELPER
# ---------------------------
def run_headless(cmd, timeout=None):
    """Run subprocess fully headless (Windows safe)"""
    creationflags = 0
    startupinfo = None

    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=creationflags,
        startupinfo=startupinfo
    )

# ---------------------------
# AI UPSCALER CLASS
# ---------------------------
class AIUpscaler:
    """Real-ESRGAN AI upscaler for video frames"""
    
    def __init__(self, gpu_id=0):
        self.upsampler = None
        self.gpu_id = gpu_id
        
        if not AI_UPSCALING_AVAILABLE:
            raise RuntimeError("AI upscaling libraries not installed")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. AI upscaling requires NVIDIA GPU.")
        
        print(f"Initializing Real-ESRGAN AI upscaler (GPU {gpu_id})...")
        
        # Download model if needed (will be cached)
        model_name = 'RealESRGAN_x4plus'  # 4x upscaling
        model_path = self._get_model_path(model_name)
        
        # Initialize model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,  # No tiling for small clips
            tile_pad=10,
            pre_pad=0,
            half=True,  # FP16 for speed on modern GPUs
            gpu_id=gpu_id
        )
        
        print("✓ AI Upscaler ready!")
    
    def _get_model_path(self, model_name):
        """Get or download model weights"""
        model_dir = os.path.join(tempfile.gettempdir(), 'realesrgan_models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'{model_name}.pth')
        
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} model (first time only)...")
            url = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth'
            
            try:
                import urllib.request
                urllib.request.urlretrieve(url, model_path)
                print(f"✓ Model downloaded to {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
        
        return model_path
    
    def upscale_image(self, img):
        """Upscale a single image (numpy array)"""
        try:
            output, _ = self.upsampler.enhance(img, outscale=4)
            return output
        except Exception as e:
            print(f"⚠ AI upscaling failed: {e}, falling back to original")
            return img
    
    def upscale_video_file(self, input_path, output_path, progress_callback=None):
        """Upscale a video file frame by frame"""
        try:
            # Open video
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get first frame to determine output size
            ret, first_frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read video")
            
            upscaled_first = self.upscale_image(first_frame)
            height, width = upscaled_first.shape[:2]
            
            # Reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create temp file for frames
            temp_frames_dir = tempfile.mkdtemp()
            
            try:
                # Process all frames
                frame_num = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # AI upscale
                    upscaled = self.upscale_image(frame)
                    
                    # Save frame
                    frame_path = os.path.join(temp_frames_dir, f'frame_{frame_num:06d}.png')
                    cv2.imwrite(frame_path, upscaled)
                    
                    frame_num += 1
                    
                    if progress_callback:
                        progress_callback(frame_num, total_frames)
                
                cap.release()
                
                # Reassemble video with FFmpeg
                ffmpeg_path = find_ffmpeg()
                if not ffmpeg_path:
                    ffmpeg_path = 'ffmpeg'
                
                cmd = [
                    ffmpeg_path,
                    '-framerate', str(fps),
                    '-i', os.path.join(temp_frames_dir, 'frame_%06d.png'),
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-y',
                    output_path
                ]
                
                result = run_headless(cmd)
                
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")
                
                return True
                
            finally:
                # Clean up temp frames
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"✗ AI video upscaling failed: {e}")
            return False

# Global upscaler instance (lazy init)
_global_upscaler = None

def get_ai_upscaler(gpu_id=0):
    """Get or create global AI upscaler instance"""
    global _global_upscaler
    if _global_upscaler is None and AI_UPSCALING_AVAILABLE:
        try:
            _global_upscaler = AIUpscaler(gpu_id=gpu_id)
        except Exception as e:
            print(f"✗ Failed to initialize AI upscaler: {e}")
            return None
    return _global_upscaler

# ---------------------------
# PROGRESS WINDOW
# ---------------------------
class ProgressWindow:
    def __init__(self, parent, total_clips):
        self.window = tk.Toplevel(parent)
        self.window.title("Generating Slideshow...")
        self.window.geometry("650x420")
        self.window.resizable(False, False)
        
        # Make window visible and on top
        self.window.deiconify()
        self.window.lift()
        self.window.attributes('-topmost', True)
        self.window.after(200, lambda: self.window.attributes('-topmost', False))
        
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)

        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.pack(fill="both", expand=True)

        ttk.Label(main_frame, text="Rendering Slideshow",
                  font=('Arial', 14, 'bold')).pack(pady=(0, 10))

        # -------- Render Info Box --------
        self.info_text = tk.Text(main_frame, height=8, width=75)
        self.info_text.pack(pady=(0, 10))
        self.info_text.config(state="disabled")

        # -------- Status --------
        self.status_label = ttk.Label(main_frame, text="Initializing...",
                                      font=('Arial', 10))
        self.status_label.pack(pady=(5, 5))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame,
                                            variable=self.progress_var,
                                            maximum=100,
                                            length=550,
                                            mode='determinate')
        self.progress_bar.pack(pady=(5, 5))

        self.percent_label = ttk.Label(main_frame, text="0%",
                                       font=('Arial', 12, 'bold'))
        self.percent_label.pack(pady=(5, 5))

        self.detail_label = ttk.Label(main_frame, text="",
                                      font=('Arial', 9),
                                      foreground='gray')
        self.detail_label.pack(pady=(5, 5))

        self.clips_label = ttk.Label(main_frame,
                                     text=f"0 / {total_clips} clips processed",
                                     font=('Arial', 9))
        self.clips_label.pack()

        self.total_clips = total_clips
        
        # Force window to display NOW
        self.window.update_idletasks()
        self.window.update()

    def set_render_info(self, info_text):
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, info_text)
        self.info_text.config(state="disabled")
        self.window.update_idletasks()
        self.window.update()

    def update_progress(self, completed, total, status_message="", detail=""):
        percentage = (completed / total * 100) if total > 0 else 0
        self.progress_var.set(percentage)
        self.percent_label.config(text=f"{percentage:.1f}%")
        self.clips_label.config(text=f"{completed} / {total} clips processed")

        if status_message:
            self.status_label.config(text=status_message)

        if detail:
            self.detail_label.config(text=detail)

        self.window.update_idletasks()
        self.window.update()

    def set_status(self, message, detail=""):
        self.status_label.config(text=message)
        if detail:
            self.detail_label.config(text=detail)
        self.window.update_idletasks()
        self.window.update()

    def close(self):
        self.window.destroy()

# ---------------------------
# GUI CONFIG DIALOG
# ---------------------------
class ConfigDialog:
    def __init__(self):
        self.result = None
        self.root = tk.Tk()
        self.root.title("Random Slideshow Generator")
        self.root.geometry("500x500")
        self.root.resizable(False, False)

        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0)

        ttk.Label(main_frame, text="Random Slideshow Generator", 
                 font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0,15))

        # Bin Name
        ttk.Label(main_frame, text="Bin Name:").grid(row=1, column=0, sticky="w", pady=8)
        self.bin_entry = ttk.Entry(main_frame, width=30)
        self.bin_entry.grid(row=1, column=1, pady=8)
        self.bin_entry.insert(0, "JJK")

        # Interval
        ttk.Label(main_frame, text="Interval (seconds):").grid(row=2, column=0, sticky="w", pady=8)
        self.interval_entry = ttk.Entry(main_frame, width=30)
        self.interval_entry.grid(row=2, column=1, pady=8)
        self.interval_entry.insert(0, "0.05")

        # Total Duration
        ttk.Label(main_frame, text="Total Duration (seconds):").grid(row=3, column=0, sticky="w", pady=8)
        self.duration_entry = ttk.Entry(main_frame, width=30)
        self.duration_entry.grid(row=3, column=1, pady=8)
        self.duration_entry.insert(0, "10")

        # FPS
        ttk.Label(main_frame, text="FPS:").grid(row=4, column=0, sticky="w", pady=8)
        self.fps_entry = ttk.Entry(main_frame, width=30)
        self.fps_entry.grid(row=4, column=1, pady=8)
        self.fps_entry.insert(0, "60")

        # Resolution
        ttk.Label(main_frame, text="Resolution:").grid(row=5, column=0, sticky="w", pady=8)
        res_frame = ttk.Frame(main_frame)
        res_frame.grid(row=5, column=1, pady=8, sticky="w")
        self.width_entry = ttk.Entry(res_frame, width=10)
        self.width_entry.pack(side="left")
        self.width_entry.insert(0, "1920")
        ttk.Label(res_frame, text=" x ").pack(side="left")
        self.height_entry = ttk.Entry(res_frame, width=10)
        self.height_entry.pack(side="left")
        self.height_entry.insert(0, "1080")

        # Video Track
        ttk.Label(main_frame, text="Import to Track:").grid(row=6, column=0, sticky="w", pady=8)
        self.track_entry = ttk.Entry(main_frame, width=30)
        self.track_entry.grid(row=6, column=1, pady=8)
        self.track_entry.insert(0, "1")

        # Parallel Jobs
        ttk.Label(main_frame, text="Parallel Jobs:").grid(row=7, column=0, sticky="w", pady=8)
        self.jobs_entry = ttk.Entry(main_frame, width=30)
        self.jobs_entry.grid(row=7, column=1, pady=8)
        self.jobs_entry.insert(0, "4")

        # Upscaling method
        ttk.Label(main_frame, text="Upscaling Method:").grid(row=8, column=0, sticky="w", pady=8)
        self.upscale_method = tk.StringVar(value="lanczos")
        method_frame = ttk.Frame(main_frame)
        method_frame.grid(row=8, column=1, sticky="w", pady=8)
        
        ttk.Radiobutton(method_frame, text="None", variable=self.upscale_method, 
                       value="none").pack(side="left", padx=5)
        ttk.Radiobutton(method_frame, text="Lanczos", variable=self.upscale_method, 
                       value="lanczos").pack(side="left", padx=5)
        
        ai_state = "normal" if AI_UPSCALING_AVAILABLE else "disabled"
        ai_radio = ttk.Radiobutton(method_frame, text="AI (GPU)", variable=self.upscale_method, 
                                   value="ai", state=ai_state)
        ai_radio.pack(side="left", padx=5)
        
        if not AI_UPSCALING_AVAILABLE:
            ttk.Label(method_frame, text="(install libs)", foreground="red", 
                     font=('Arial', 8)).pack(side="left")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=9, column=0, columnspan=2, pady=(15,0))
        ttk.Button(button_frame, text="Generate & Import", command=self.ok).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=5)

        self.root.bind("<Return>", lambda e: self.ok())
        self.root.bind("<Escape>", lambda e: self.cancel())
        self.bin_entry.focus()

    def ok(self):
        try:
            bin_name = self.bin_entry.get().strip()
            interval = float(self.interval_entry.get())
            duration = float(self.duration_entry.get())
            fps = int(self.fps_entry.get())
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())
            track = int(self.track_entry.get())
            jobs = int(self.jobs_entry.get())
            upscale_method = self.upscale_method.get()
            
            if not bin_name or interval <= 0 or duration <= 0 or fps <= 0 or jobs <= 0:
                raise ValueError("Invalid values")
            
            if upscale_method == "ai" and not AI_UPSCALING_AVAILABLE:
                messagebox.showerror("Error", "AI upscaling libraries not installed!")
                return
                
            self.result = (bin_name, interval, duration, fps, width, height, track, jobs, upscale_method)
            # Withdraw instead of destroy so we can use it for progress window
            self.root.withdraw()
            self.root.quit()  # Exit mainloop but keep window
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def cancel(self):
        self.result = None
        self.root.quit()
        self.root.destroy()

# ---------------------------
# CLIP SELECTION MANAGER
# ---------------------------
class ClipSelector:
    """Manages clip selection to avoid duplicates until all clips are used"""
    def __init__(self, clip_paths):
        self.all_clips = clip_paths.copy()
        self.unused_clips = clip_paths.copy()
        self.used_clips = []
        random.shuffle(self.unused_clips)
    
    def get_next_clip(self):
        """Get next clip, resetting pool if all clips have been used"""
        if not self.unused_clips:
            print("  → All clips used! Resetting pool...")
            self.unused_clips = self.all_clips.copy()
            self.used_clips = []
            random.shuffle(self.unused_clips)
        
        clip = self.unused_clips.pop()
        self.used_clips.append(clip)
        return clip
    
    def get_stats(self):
        """Get current usage statistics"""
        return {
            'total': len(self.all_clips),
            'used': len(self.used_clips),
            'remaining': len(self.unused_clips)
        }

# ---------------------------
# BIN SEARCH
# ---------------------------
def find_bin_recursive(folder, bin_name):
    if folder.GetName() == bin_name:
        return folder
    subfolders = folder.GetSubFolderList()
    if subfolders:
        for subfolder in subfolders:
            found = find_bin_recursive(subfolder, bin_name)
            if found:
                return found
    return None

def get_clip_paths_from_bin(media_pool, bin_name):
    """Get file paths of all clips in the specified bin"""
    root_folder = media_pool.GetRootFolder()
    bin_folder = find_bin_recursive(root_folder, bin_name)
    
    if not bin_folder:
        return []
    
    clips = bin_folder.GetClipList()
    clip_paths = []
    
    for clip in clips:
        file_path = clip.GetClipProperty("File Path")
        if file_path and os.path.exists(file_path):
            clip_paths.append(file_path)
    
    return clip_paths

# ---------------------------
# FFMPEG UTILITIES
# ---------------------------
def find_ffmpeg():
    """Find FFmpeg executable - check Resolve installation first"""
    
    # Common FFmpeg locations with Resolve
    resolve_paths = [
        r"C:\Program Files\Blackmagic Design\DaVinci Resolve\ffmpeg.exe",
        r"C:\Program Files\Blackmagic Design\DaVinci Resolve\libs\ffmpeg.exe",
        "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Frameworks/ffmpeg",
        "/opt/resolve/libs/ffmpeg"
    ]
    
    # Check Resolve paths first
    for path in resolve_paths:
        if os.path.exists(path):
            return path
    
    # Try system PATH (headless)
    try:
        result = run_headless(['ffmpeg', '-version'])
        if result.returncode == 0:
            return 'ffmpeg'
    except:
        pass
    
    return None

# Duration cache to avoid repeated ffprobe calls
duration_cache = {}
dimension_cache = {}

def get_media_duration(ffmpeg_path, media_file):
    """Get duration of a media file using ffprobe (with caching)"""
    if media_file in duration_cache:
        return duration_cache[media_file]
    
    try:
        # Try ffprobe first (more reliable)
        ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
        
        cmd = [
            ffprobe_path if os.path.exists(ffprobe_path) else 'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            media_file
        ]
        
        result = run_headless(cmd)
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            duration_cache[media_file] = duration
            return duration
    except:
        pass
    
    # Fallback: assume videos are long enough
    duration_cache[media_file] = 10.0
    return 10.0

def get_media_dimensions(ffmpeg_path, media_file):
    """Get width and height of a media file using ffprobe (with caching)"""
    if media_file in dimension_cache:
        return dimension_cache[media_file]
    
    try:
        ffprobe_path = ffmpeg_path.replace('ffmpeg', 'ffprobe')
        
        cmd = [
            ffprobe_path if os.path.exists(ffprobe_path) else 'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            media_file
        ]
        
        result = run_headless(cmd)
        if result.returncode == 0 and result.stdout.strip():
            width, height = map(int, result.stdout.strip().split(','))
            dimension_cache[media_file] = (width, height)
            return (width, height)
    except:
        pass
    
    # Fallback: assume HD
    dimension_cache[media_file] = (1920, 1080)
    return (1920, 1080)

def is_video_file(filepath):
    """Check if file is a video (including GIFs)"""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.gif', '.m4v', '.flv', '.wmv'}
    return os.path.splitext(filepath)[1].lower() in video_extensions

def is_image_file(filepath):
    """Check if file is a static image"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return os.path.splitext(filepath)[1].lower() in image_extensions

def get_random_interval(base_interval, variation_percent=0.10):
    """Return interval randomly varied by ±variation_percent"""
    min_interval = base_interval * (1 - variation_percent)
    max_interval = base_interval * (1 + variation_percent)
    return random.uniform(min_interval, max_interval)

# ---------------------------
# PARALLEL CLIP PROCESSING
# ---------------------------
def process_single_clip(args):
    """Process a single clip - designed to run in parallel"""
    i, media_file, interval, fps, width, height, temp_dir, ffmpeg_path, upscale_method = args
    
    temp_output = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
    
    try:
        # Get input dimensions for upscaling decision
        input_width, input_height = get_media_dimensions(ffmpeg_path, media_file)
        needs_upscale = upscale_method != "none" and (input_width < width or input_height < height)
        
        # Check if it's a video/GIF or image
        if is_video_file(media_file):
            # Video or GIF - extract random segment first
            duration = get_media_duration(ffmpeg_path, media_file)
            
            if duration > interval:
                max_start = duration - interval
                start_time = random.uniform(0, max_start)
            else:
                start_time = 0
            
            # Extract segment to temp file
            temp_segment = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
            
            extract_cmd = [
                ffmpeg_path,
                '-ss', str(start_time),
                '-i', media_file,
                '-t', str(interval),
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-an',
                '-y',
                temp_segment
            ]
            
            result = run_headless(extract_cmd, timeout=30)
            if result.returncode != 0:
                return (i, None, f"✗ {os.path.basename(media_file)}: Extract failed")
            
            # AI upscale if needed
            if needs_upscale and upscale_method == "ai":
                upscaler = get_ai_upscaler()
                if upscaler:
                    temp_upscaled = os.path.join(temp_dir, f"upscaled_{i:04d}.mp4")
                    success = upscaler.upscale_video_file(temp_segment, temp_upscaled)
                    
                    if success:
                        os.remove(temp_segment)
                        temp_segment = temp_upscaled
                        upscale_note = " [AI-UPSCALED]"
                    else:
                        upscale_note = " [AI-FAILED, FALLBACK]"
                        needs_upscale = True  # Fall back to Lanczos
                        upscale_method = "lanczos"
                else:
                    upscale_note = " [AI-UNAVAILABLE]"
                    needs_upscale = True
                    upscale_method = "lanczos"
            else:
                upscale_note = ""
            
            # Final resize and padding
            if needs_upscale and upscale_method == "lanczos":
                vf = f'scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}'
                upscale_note = " [LANCZOS]"
            else:
                vf = f'fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
            
            final_cmd = [
                ffmpeg_path,
                '-i', temp_segment,
                '-vf', vf,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-an',
                '-y',
                temp_output
            ]
            
            result = run_headless(final_cmd, timeout=30)
            
            # Clean up temp segment
            try:
                os.remove(temp_segment)
            except:
                pass
            
            if result.returncode == 0:
                return (i, temp_output, f"✓ {os.path.basename(media_file)}{upscale_note}")
            else:
                return (i, None, f"✗ {os.path.basename(media_file)}: Processing failed")
        
        elif is_image_file(media_file):
            # Static image - AI upscale if needed
            upscale_note = ""
            source_image = media_file
            
            if needs_upscale and upscale_method == "ai":
                upscaler = get_ai_upscaler()
                if upscaler:
                    try:
                        img = cv2.imread(media_file)
                        upscaled_img = upscaler.upscale_image(img)
                        
                        temp_upscaled_img = os.path.join(temp_dir, f"upscaled_img_{i:04d}.png")
                        cv2.imwrite(temp_upscaled_img, upscaled_img)
                        source_image = temp_upscaled_img
                        upscale_note = " [AI-UPSCALED]"
                    except Exception as e:
                        print(f"AI upscale failed for {media_file}: {e}")
                        upscale_note = " [AI-FAILED, FALLBACK]"
                        upscale_method = "lanczos"
            
            # Create video from image
            if upscale_method == "lanczos" and needs_upscale:
                vf = f'scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}'
                if not upscale_note:
                    upscale_note = " [LANCZOS]"
            else:
                vf = f'fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
            
            cmd = [
                ffmpeg_path,
                '-loop', '1',
                '-i', source_image,
                '-t', str(interval),
                '-vf', vf,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-y',
                temp_output
            ]
            
            result = run_headless(cmd, timeout=30)
            
            # Clean up temp upscaled image
            if source_image != media_file:
                try:
                    os.remove(source_image)
                except:
                    pass
            
            if result.returncode == 0:
                return (i, temp_output, f"✓ {os.path.basename(media_file)}{upscale_note}")
            else:
                return (i, None, f"✗ {os.path.basename(media_file)}: Failed")
        
        else:
            return (i, None, f"Unsupported file type: {os.path.basename(media_file)}")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (i, None, f"✗ {os.path.basename(media_file)}: {str(e)}")

# ---------------------------
# VIDEO GENERATION WITH FFMPEG (OPTIMIZED)
# ---------------------------
def create_slideshow_ffmpeg(clip_paths, interval, total_duration, fps, width, height, output_path, max_workers=4, upscale_method="lanczos", progress_window=None):
    """Create slideshow video using FFmpeg with parallel processing"""
    
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        messagebox.showerror("Error", 
            "FFmpeg not found!\n\n"
            "Please install FFmpeg or ensure DaVinci Resolve is properly installed.")
        return False
    
    # Initialize AI upscaler if needed
    if upscale_method == "ai":
        print("\nInitializing AI upscaler...")
        upscaler = get_ai_upscaler()
        if not upscaler:
            messagebox.showerror("Error", "Failed to initialize AI upscaler!")
            return False
    
    print(f"\n{'='*60}")
    print(f"GENERATING SLIDESHOW VIDEO (OPTIMIZED & HEADLESS)")
    print(f"{'='*60}")
    print(f"FFmpeg: {ffmpeg_path}")
    print(f"Available clips: {len(clip_paths)}")
    print(f"Interval: {interval}s")
    print(f"Total duration: {total_duration}s")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Parallel workers: {max_workers}")
    print(f"Upscaling: {upscale_method.upper()}")
    print(f"{'='*60}\n")
    
    if progress_window:
        render_summary = (
            f"FFmpeg: {ffmpeg_path}\n"
            f"Available clips: {len(clip_paths)}\n"
            f"Interval: {interval}s\n"
            f"Total duration: {total_duration}s\n"
            f"Output: {os.path.basename(output_path)}\n"
            f"FPS: {fps}\n"
            f"Resolution: {width}x{height}\n"
            f"Parallel workers: {max_workers}\n"
            f"Upscaling: {upscale_method.upper()}"
        )

        progress_window.set_render_info(render_summary)
        progress_window.set_status("Preparing clips...")

    # Calculate number of clips needed
    num_clips = int(total_duration / interval)
    print(f"Will create {num_clips} clips at {interval}s each\n")
    
    if progress_window:
        progress_window.set_status("Preparing clips...", f"Total: {num_clips} clips")
    
    # Initialize clip selector to prevent duplicates
    clip_selector = ClipSelector(clip_paths)
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Prepare clip selection with random interval variance
        selected_clips = []
        for i in range(num_clips):
            media_file = clip_selector.get_next_clip()
            varied_interval = get_random_interval(interval, variation_percent=0.10)  # ±10% variance
            selected_clips.append((i, media_file, varied_interval, fps, width, height, temp_dir, ffmpeg_path, upscale_method))

        
        # Show selection stats
        stats = clip_selector.get_stats()
        print(f"Clip Selection Stats:")
        print(f"  Total unique clips: {stats['total']}")
        print(f"  Clips used: {stats['used']}")
        print(f"  Remaining: {stats['remaining']}")
        print(f"\n{'='*60}\n")
        
        if progress_window:
            progress_window.set_status("Processing clips...", 
                                      f"Using {max_workers} parallel workers")
        
        # Process clips in parallel
        processed_files = {}
        completed = 0
        
        print(f"Processing {num_clips} clips with {max_workers} parallel workers...\n")
        
        # For AI upscaling, use fewer workers to avoid VRAM issues
        if upscale_method == "ai":
            max_workers = min(max_workers, 2)
            print(f"⚠ AI upscaling: reducing workers to {max_workers} to conserve GPU memory\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = {executor.submit(process_single_clip, args): args[0] for args in selected_clips}
            
            # Collect results as they complete
            for future in as_completed(futures):
                i, output_file, message = future.result()
                print(f"[{i+1}/{num_clips}] {message}")
                
                if output_file:
                    processed_files[i] = output_file
                    completed += 1
                    
                    if progress_window:
                        # Extract just the filename from the message for display
                        clip_name = message.replace("✓ ", "").split(":")[0]
                        progress_window.update_progress(completed, num_clips, 
                                                       "Processing clips...",
                                                       f"Last: {clip_name}")
        
        if not processed_files:
            messagebox.showerror("Error", "No clips were processed successfully")
            return False
        
        # Sort by index to maintain order
        sorted_files = [processed_files[i] for i in sorted(processed_files.keys())]
        
        print(f"\n{'='*60}")
        print(f"Concatenating {len(sorted_files)} clips...")
        print(f"{'='*60}\n")
        
        if progress_window:
            progress_window.set_status("Concatenating clips...", 
                                      f"Merging {len(sorted_files)} clips into final video")
        
        # Create concat file
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for pf in sorted_files:
                # Escape path for FFmpeg
                escaped_path = pf.replace('\\', '/').replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
        
        # Concatenate all clips (headless)
        concat_cmd = [
            ffmpeg_path,
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',  # FASTER: stream copy, no re-encoding
            '-y',
            output_path
        ]
        
        result = run_headless(concat_cmd)
        
        if result.returncode == 0:
            print(f"\n{'='*60}")
            print(f"✓ SUCCESS! Video created: {output_path}")
            print(f"{'='*60}\n")
            
            if progress_window:
                progress_window.update_progress(num_clips, num_clips, 
                                               "✓ Video created successfully!",
                                               "Ready to import to timeline")
            return True
        else:
            print(f"\n✗ Concatenation failed: {result.stderr}")
            messagebox.showerror("Error", f"Failed to concatenate clips:\n{result.stderr[:500]}")
            return False
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"Failed to create video:\n{e}")
        return False
    
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

# ---------------------------
# IMPORT TO RESOLVE
# ---------------------------
def import_to_resolve_timeline(video_path, track_number, progress_window=None):
    """Import generated video to Resolve timeline"""
    try:
        if progress_window:
            progress_window.set_status("Importing to Resolve...", "Connecting to Resolve")
        
        resolve = app.GetResolve()
        if not resolve:
            print("✗ Resolve not available")
            return False
        
        project = resolve.GetProjectManager().GetCurrentProject()
        if not project:
            print("✗ No project open")
            return False
        
        timeline = project.GetCurrentTimeline()
        if not timeline:
            print("✗ No active timeline")
            return False
        
        media_pool = project.GetMediaPool()
        media_storage = resolve.GetMediaStorage()
        
        print(f"\nImporting to Resolve...")
        print(f"Video: {video_path}")
        print(f"File exists: {os.path.exists(video_path)}")
        print(f"File size: {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes")
        
        if progress_window:
            progress_window.set_status("Importing to Resolve...", "Adding to Media Pool")
        
        # Import video to media pool
        imported = media_storage.AddItemListToMediaPool([video_path])
        
        print(f"Import result: {imported}")
        
        if not imported:
            print("✗ Failed to import to Media Pool")
            print("Trying alternative import method...")
            
            # Alternative: Try using SubClipMediaPool
            try:
                current_folder = media_pool.GetCurrentFolder()
                imported_clips = media_pool.ImportMedia([video_path])
                print(f"Alternative import result: {imported_clips}")
                
                if not imported_clips or len(imported_clips) == 0:
                    print("✗ Alternative import also failed")
                    return False
            except Exception as e:
                print(f"✗ Alternative import error: {e}")
                return False
        
        print(f"✓ Imported to Media Pool")
        
        if progress_window:
            progress_window.set_status("Importing to Resolve...", "Finding imported clip")
        
        # Find the imported clip - try current folder first
        current_folder = media_pool.GetCurrentFolder()
        clips = current_folder.GetClipList()
        
        target_clip = None
        
        # First try: Check current folder
        print(f"Searching in current folder: {current_folder.GetName()}")
        for clip in clips:
            clip_path = clip.GetClipProperty("File Path")
            if clip_path:
                print(f"  Found clip: {os.path.basename(clip_path)}")
                if os.path.normpath(clip_path) == os.path.normpath(video_path):
                    target_clip = clip
                    print(f"  ✓ Match found!")
                    break
        
        # Second try: Check root folder if not found
        if not target_clip:
            print("Not found in current folder, checking root...")
            root_folder = media_pool.GetRootFolder()
            clips = root_folder.GetClipList()
            
            for clip in clips:
                clip_path = clip.GetClipProperty("File Path")
                if clip_path and os.path.normpath(clip_path) == os.path.normpath(video_path):
                    target_clip = clip
                    print(f"  ✓ Found in root folder!")
                    break
        
        if not target_clip:
            print("✗ Could not find imported clip in media pool")
            print("The file was imported but couldn't be located. Please add to timeline manually.")
            return False
        
        if progress_window:
            progress_window.set_status("Importing to Resolve...", "Adding to timeline")
        
        # Add to timeline
        print(f"Adding to timeline...")
        
        # Try to append to timeline
        result = media_pool.AppendToTimeline([target_clip])
        
        print(f"AppendToTimeline result: {result}")
        
        if result:
            print(f"✓ Added to timeline!")
            if progress_window:
                progress_window.set_status("✓ Import Complete!", "Added to timeline")
            
            messagebox.showinfo("Success", 
                f"✓ Slideshow created and imported!\n\n"
                f"File: {os.path.basename(video_path)}\n"
                f"Added to timeline")
            return True
        else:
            print("✗ Failed to add to timeline")
            print("Trying alternative method...")
            
            # Alternative: Try using different API call
            try:
                # Set current folder
                media_pool.SetCurrentFolder(root_folder if target_clip in root_folder.GetClipList() else current_folder)
                
                # Try adding with position
                result = media_pool.AppendToTimeline(target_clip)
                
                if result:
                    print(f"✓ Alternative method succeeded!")
                    messagebox.showinfo("Success", 
                        f"✓ Slideshow created and imported!\n\n"
                        f"File: {os.path.basename(video_path)}\n"
                        f"Added to timeline")
                    return True
            except Exception as e:
                print(f"Alternative method error: {e}")
            
            messagebox.showwarning("Partial Success", 
                f"Video created and imported to Media Pool:\n{os.path.basename(video_path)}\n\n"
                f"But could not add to timeline automatically.\n"
                f"Please drag it to the timeline manually from the Media Pool.")
            return False
            
    except Exception as e:
        print(f"✗ Error importing to Resolve: {e}")
        import traceback
        traceback.print_exc()
        return False

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def main():
    print("\n" + "="*60)
    print("RANDOM MEDIA SLIDESHOW GENERATOR (AI UPSCALING)")
    print("="*60 + "\n")
    
    # Get Resolve context
    try:
        resolve = app.GetResolve()
        if not resolve:
            messagebox.showerror("Error", "Cannot access DaVinci Resolve")
            return
        
        project = resolve.GetProjectManager().GetCurrentProject()
        if not project:
            messagebox.showerror("Error", "No project open")
            return
        
        media_pool = project.GetMediaPool()
        
    except Exception as e:
        messagebox.showerror("Error", f"Resolve error: {e}")
        return
    
    # Show config dialog
    dialog = ConfigDialog()
    dialog.root.mainloop()  # Start event loop
    
    # After mainloop exits, check if we have a result
    if dialog.result is None:
        print("Cancelled by user.")
        try:
            dialog.root.destroy()
        except:
            pass
        return
    
    bin_name, interval, duration, fps, width, height, track, max_workers, upscale_method = dialog.result
    
    # Get clip paths from bin
    print(f"Searching for bin: {bin_name}")
    clip_paths = get_clip_paths_from_bin(media_pool, bin_name)
    
    if not clip_paths:
        messagebox.showerror("Error", f"No clips found in bin '{bin_name}'")
        dialog.root.destroy()
        return
    
    print(f"Found {len(clip_paths)} clips in bin '{bin_name}'")
    
    # Generate output path in temp directory
    temp_dir = tempfile.gettempdir()
    output_filename = f"slideshow_{bin_name}_{interval}s.mp4"
    output_path = os.path.join(temp_dir, output_filename)
    
    print(f"Output will be: {output_path}")
    
    # Calculate total clips needed
    num_clips = int(duration / interval)
    
    # Confirm
    confirm = messagebox.askyesno("Confirm",
        f"Generate slideshow from bin '{bin_name}'?\n\n"
        f"Clips in bin: {len(clip_paths)}\n"
        f"Interval: {interval}s\n"
        f"Duration: {duration}s\n"
        f"Total clips to render: {num_clips}\n"
        f"Output: {output_filename}\n"
        f"Parallel workers: {max_workers}\n"
        f"Upscaling: {upscale_method.upper()}\n"
        f"Will auto-import to track {track}")
    
    if not confirm:
        dialog.root.destroy()
        return
    
    # Create progress window
    progress = ProgressWindow(dialog.root, num_clips)
    
    # Generate video with progress updates
    success = create_slideshow_ffmpeg(
        clip_paths, interval, duration, fps, width, height, output_path, max_workers,
        upscale_method=upscale_method,
        progress_window=progress
    )
    
    if not success:
        progress.close()
        dialog.root.destroy()
        messagebox.showerror("Error", "Failed to generate slideshow video")
        return
    
    # Update progress for import phase
    progress.set_status("Importing to Resolve...", "Adding to Media Pool and Timeline")
    
    # Import to Resolve
    import_success = import_to_resolve_timeline(output_path, track, progress_window=progress)
    
    # Close everything
    progress.close()
    dialog.root.destroy()
    
    if not import_success:
        messagebox.showwarning("Partial Success",
            f"Video created successfully but failed to import:\n{output_path}\n\n"
            f"Please import manually.")


if __name__ == "__main__":
    main()

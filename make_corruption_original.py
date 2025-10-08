# pip install opencv-python h5py numpy scikit-image tqdm

import os
import cv2
import h5py
import numpy as np
import subprocess
import json
from pathlib import Path
import tempfile
import skimage as sk
from skimage.color import hsv2rgb, rgb2hsv
from tqdm import tqdm
import multiprocessing
import random
from collections import defaultdict


SOURCE_DIR = Path("/INPUT")

OUTPUT_DIR = Path("/OUTPUT")

APPLY_CORRUPTIONS = {
    "shot_noise": False, "rain": False, "contrast": False, "brightness": False,
    "saturate": False, "fog": False,
    "motion_blur": False,
    "bit_error": True, "h264_crf": False, "h264_abr": False, "h265_crf": False,
    "h265_abr": False, "frame_rate": False,
}

APPLY_PARTIAL_CORRUPTION = False


FrameBasedCorruption = {"shot_noise", "rain", "contrast", "brightness", "saturate", "fog"}
VideoBasedCorruption = {"bit_error", "h264_crf", "h264_abr", "h265_crf", "h265_abr", "frame_rate", "motion_blur"}


FIXED_LABEL = 1
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.webm']
NUM_WORKERS = 0
NUM_FRAMES_TO_EXTRACT = 32
FRAME_STRIDE = 8
TARGET_RESOLUTION = (256, 256)


def resize_frames(frames, size):

    if not frames:
        return []

    return [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]

def load_video_frames(video_path, num_frames=32, stride=8):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f" [Error] Could not open video: {video_path}")
        return None
    
    frames = []
    

    if num_frames == float('inf'):

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    else:

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(num_frames):
            frame_idx = i * stride
            if frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break

    cap.release()
    return frames if frames else None

def save_videos_to_hdf5_group(video_data_list, label, dst_path):

    if not video_data_list: return
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst_path, 'w') as hf:
            for video_data in tqdm(video_data_list, desc=f"Saving to {dst_path.name}"):
                video_name = video_data['name']
                frames = video_data['frames']
                if not frames: continue
                
                video_group = hf.create_group(video_name)
                video_group.create_dataset('video', data=np.array(frames, dtype=np.uint8), compression='gzip')
                video_group.create_dataset('label', data=int(label))
    except Exception as e:
        print(f" [Error] HDF5 Group Save Failed for {dst_path}: {e}")


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255
def rain(x, severity=1): return x # Placeholder
def contrast(x, severity=1):
    c = [0.5, 0.4, .3, .2, .1][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255
def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = hsv2rgb(x)
    return np.clip(x, 0, 1) * 255
def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = hsv2rgb(x)
    return np.clip(x, 0, 1) * 255
def motion_blur(frames, severity):
    c = [1, 2, 3, 4, 6][severity-1]
    clip = np.asarray(frames, dtype=np.float32)
    blur_clip = []
    for i in range(len(clip)):
        s = slice(max(0, i - c), min(len(clip), i + c + 1))
        blur_image = np.mean(clip[s], axis=0)
        blur_clip.append(np.array(blur_image, dtype=np.uint8))
    return blur_clip
def plasma_fractal(mapsize=256, wibbledecay=3): # Helper for fog
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100
    def wibbledmean(array): return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)
    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)
    def filldiamonds():
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)
    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay
    maparray -= maparray.min()
    return maparray / maparray.max()
def fog(frames, severity):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    fog_clip = []
    for image in frames:
        image_arr = np.array(image) / 255.
        fog_layer = c[0] * plasma_fractal(wibbledecay=c[1])
        h, w, _ = image_arr.shape
        fog_layer_resized = cv2.resize(fog_layer, (w, h))
        image_arr += np.repeat(fog_layer_resized[..., np.newaxis], 3, axis=2)
        max_val = np.max(image_arr)
        image_arr = image_arr * max_val / (max_val + c[0])
        fog_image = np.array(np.clip(image_arr, 0, 1) * 255, dtype=np.uint8)
        fog_clip.append(fog_image)
    return fog_clip
def bit_error(src, dst, severity):
    c = [100000, 50000, 30000, 20000, 10000][severity-1]
    return subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c", "copy", "-bsf:v", f"noise={c}", str(dst)]).returncode
def h264_crf(src, dst, severity):
    c = [23, 30, 37, 44, 51][severity-1]
    return subprocess.call(["ffmpeg", "-y", "-i", str(src), "-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-crf", str(c), str(dst)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def h264_abr(src, dst, severity):
    c = [2, 4, 8, 16, 32][severity-1]
    result = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(src)], capture_output=True, text=True)
    data = json.loads(result.stdout)
    bit_rate = str(int(float(data['format']['bit_rate']) / c))
    return subprocess.call(["ffmpeg", "-y", "-i", str(src), "-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize", bit_rate, str(dst)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def h265_crf(src, dst, severity):
    c = [27, 33, 39, 45, 51][severity - 1]
    return subprocess.call(["ffmpeg", "-y", "-i", str(src),"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-crf", str(c), str(dst)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def h265_abr(src, dst, severity):
    c = [2, 4, 8, 16, 32][severity - 1]
    result = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(src)], capture_output=True, text=True)
    data = json.loads(result.stdout)
    bit_rate = str(int(float(data['format']['bit_rate']) / c))
    return subprocess.call(["ffmpeg", "-y", "-i", str(src), "-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize", bit_rate, str(dst)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def frame_rate(src, dst, severity):
    c = [10, 8, 6, 4, 2][severity-1]
    return subprocess.call(["ffmpeg", "-y", "-i", str(src), "-vf", f"fps={c},crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", str(dst)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



def process_video_task(task):

    src_path, key, severity, corruption_funcs = task
    video_name = src_path.stem
    try:
        corruption_func = corruption_funcs[key]
        corrupted_frames = []
        if key in FrameBasedCorruption:
            frames = load_video_frames(src_path, NUM_FRAMES_TO_EXTRACT, FRAME_STRIDE)
            if frames is None: return None
            if key == "fog":
                corrupted_frames = corruption_func(frames, severity)
            else:
                corrupted_frames = [np.array(corruption_func(frame, severity), dtype=np.uint8) for frame in frames]
        elif key in VideoBasedCorruption:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
                tmp_path = Path(tmp_video.name)
                if key == "motion_blur":
                    full_frames = load_video_frames(src_path, num_frames=float('inf'), stride=1)
                    if full_frames is None: return None
                    corrupted_full_frames = corruption_func(full_frames, severity)
                    height, width, _ = corrupted_full_frames[0].shape
                    out = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (width, height))
                    for frame in corrupted_full_frames: out.write(frame)
                    out.release()
                else:
                    if corruption_funcs[key](src_path, tmp_path, severity) != 0: return None
                corrupted_frames = load_video_frames(tmp_path, NUM_FRAMES_TO_EXTRACT, FRAME_STRIDE)
        
        if corrupted_frames:

            resized_corrupted_frames = resize_frames(corrupted_frames, TARGET_RESOLUTION)
            return {'name': video_name, 'frames': resized_corrupted_frames}
        return None
    except Exception as e:
        print(f" [Error] Task Failed for {src_path.name}: {e}")
        return None

def process_partial_corruption_task(task):

    src_path, corruption_func = task
    video_name = src_path.stem
    severity = 3
    try:
        original_frames = load_video_frames(src_path, NUM_FRAMES_TO_EXTRACT, FRAME_STRIDE)
        if not original_frames or len(original_frames) != NUM_FRAMES_TO_EXTRACT: return None
            
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video:
            tmp_path = Path(tmp_video.name)
            if corruption_func(src_path, tmp_path, severity) != 0: return None
            corrupted_frames_base = load_video_frames(tmp_path, NUM_FRAMES_TO_EXTRACT, FRAME_STRIDE)
            if not corrupted_frames_base or len(corrupted_frames_base) != NUM_FRAMES_TO_EXTRACT: return None
        
        video_results = {}
        for num_corrupted in [8, 16, 24]:
            # Consecutive
            start_idx = random.randint(0, NUM_FRAMES_TO_EXTRACT - num_corrupted)
            final_frames_consecutive = original_frames[:]
            final_frames_consecutive[start_idx : start_idx + num_corrupted] = corrupted_frames_base[start_idx : start_idx + num_corrupted]
     
            video_results[f'consecutive_{num_corrupted}f'] = resize_frames(final_frames_consecutive, TARGET_RESOLUTION)

            # Distributed
            block_size = 4
            num_blocks = num_corrupted // block_size
            possible_start_indices = list(range(0, NUM_FRAMES_TO_EXTRACT, block_size))
            chosen_block_starts = random.sample(possible_start_indices, num_blocks)
            final_frames_distributed = original_frames[:]
            for start in chosen_block_starts:
                final_frames_distributed[start : start + block_size] = corrupted_frames_base[start : start + block_size]
         
            video_results[f'distributed_{num_corrupted}f'] = resize_frames(final_frames_distributed, TARGET_RESOLUTION)
            
        return {'name': video_name, 'results': video_results}
    except Exception as e:
        print(f" [Error] FAILED (Partial Task) for {src_path.name}: {e}")
        return None

def main():
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    video_paths_by_split = defaultdict(list)
    print("Scanning video files and grouping by split (train/val/test)...")
    for manip_folder in SOURCE_DIR.iterdir():
        if not manip_folder.is_dir(): continue
        for split_folder in manip_folder.iterdir():
            if not split_folder.is_dir(): continue
            split_name = split_folder.name.lower().replace("validation", "val").replace("valid", "val")
            if split_name in ["train", "val", "test"]:
                for ext in VIDEO_EXTENSIONS:
                    video_paths_by_split[split_name].extend(list(split_folder.glob(f'*{ext}')))

    if not video_paths_by_split:
        print(f"No video files found in train/val/test subdirectories of {SOURCE_DIR}")
        return
        
    for split, paths in video_paths_by_split.items():
        print(f"Found {len(paths)} videos for '{split}' split.")

    corruption_funcs = { "shot_noise": shot_noise, "rain": rain, "contrast": contrast, "brightness": brightness, "saturate": saturate, "motion_blur": motion_blur, "fog": fog, "bit_error": bit_error, "h264_crf": h264_crf, "h264_abr": h264_abr, "h265_crf": h265_crf, "h265_abr": h265_abr, "frame_rate": frame_rate }
    severities_to_apply = [1, 3, 5]
    workers = os.cpu_count() if NUM_WORKERS <= 0 else NUM_WORKERS


    for split_name, video_paths in video_paths_by_split.items():
        for key, is_enabled in APPLY_CORRUPTIONS.items():
            if not is_enabled: continue
            for severity in severities_to_apply:
                output_filename = f"{split_name}_{key}_{severity}.h5"
                dst_path = OUTPUT_DIR / output_filename
                if dst_path.exists():
                    print(f"Skipping, already exists: {dst_path}")
                    continue
                
                print(f"\n--- Processing Grouped: {split_name} | {key} | Severity {severity} ---")
                tasks = [(src_path, key, severity, corruption_funcs) for src_path in video_paths]
                all_processed_videos = []
                with multiprocessing.Pool(processes=workers) as pool:
                    results_iterator = pool.imap_unordered(process_video_task, tasks)
                    for result in tqdm(results_iterator, total=len(tasks), desc="Corrupting videos"):
                        if result: all_processed_videos.append(result)
                
                if all_processed_videos:
                    save_videos_to_hdf5_group(all_processed_videos, FIXED_LABEL, dst_path)
                else:
                    print(f"Warning: No videos were successfully processed for {output_filename}")


    if APPLY_PARTIAL_CORRUPTION and APPLY_CORRUPTIONS.get("bit_error", False):
        print("\n--- Processing Partial bit_error (severity 3) experiment ---")
        
        for split_name, video_paths in video_paths_by_split.items():
            print(f"\n--- Processing Partial for '{split_name}' split ---")
            tasks = [(src_path, corruption_funcs["bit_error"]) for src_path in video_paths]
            
            videos_by_partial_type = defaultdict(list)
            
            with multiprocessing.Pool(processes=workers) as pool:
                results_iterator = pool.imap_unordered(process_partial_corruption_task, tasks)
                for result in tqdm(results_iterator, total=len(tasks), desc=f"Partial corrupt ({split_name})"):
                    if result and isinstance(result, dict):
                        video_name = result['name']
                        for partial_type, frames in result['results'].items():
                            videos_by_partial_type[partial_type].append({'name': video_name, 'frames': frames})
            
            for partial_type, video_data_list in videos_by_partial_type.items():
                output_filename = f"{split_name}_{partial_type}.h5"
                dst_path = OUTPUT_DIR / output_filename
                if dst_path.exists():
                    print(f"Skipping, already exists: {dst_path}")
                    continue
                save_videos_to_hdf5_group(video_data_list, FIXED_LABEL, dst_path)
                
    print("\nAll tasks completed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
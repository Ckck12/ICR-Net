
import os
import cv2
import h5py
import numpy as np
import subprocess
import json
from pathlib import Path
import tempfile
from tqdm import tqdm
from collections import defaultdict
import shutil
import multiprocessing


BASE_DATA_DIR = Path("/dataset") 

SOURCE_DIR = Path("/INPUT")
OUTPUT_DIR = Path("/OUTPUT")


SPLIT_DIR = Path("/INPUT") 

PACKET_LOSS_REPO_PATH = Path("/packet-loss-simulation") 


SEVERITIES = [1, 3, 5]


VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.webm']
NUM_FRAMES_TO_EXTRACT = 32
FRAME_STRIDE = 8
TARGET_RESOLUTION = (256, 256)


REAL_LABEL = 0
FAKE_LABEL = 1


def resize_frames(frames, size):

    if not frames: return []
    return [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]

def load_video_frames(video_path, num_frames=32, stride=8):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f" [Error] Could not open video: {video_path}")
        return None
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(num_frames):
        frame_idx = i * stride
        if frame_idx >= total_frames: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret: frames.append(frame)
        else: break
    cap.release()
    return frames if frames else None

def save_videos_to_hdf5_group(video_data_list, dst_path):

    if not video_data_list: return
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst_path, 'w') as hf:
            for video_data in tqdm(video_data_list, desc=f"Saving to {dst_path.name}"):
                video_name, frames, label = video_data['name'], video_data['frames'], video_data['label']
                if not frames: continue
                video_group = hf.create_group(video_name)
                video_group.create_dataset('video', data=np.array(frames, dtype=np.uint8), compression='gzip')
                video_group.create_dataset('label', data=int(label))
    except Exception as e:
        print(f" [Error] HDF5 Group Save Failed for {dst_path}: {e}")


def convert_to_h264(task):

    src_path, h264_input_dir = task
    try:
        class_name = src_path.parent.name
        class_dir = h264_input_dir / class_name
        class_dir.mkdir(exist_ok=True)
        dst_h264_path = class_dir / f"{src_path.stem}.h264"
        

        if dst_h264_path.exists():
            return True

        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_path), "-vcodec", "libx264", "-an", "-f", "h264", str(dst_h264_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        print(f"Failed to convert {src_path.name}: {e}")
        return False

def get_video_label_and_id(video_path):

    parent_name = video_path.parent.name
    

    if parent_name == 'Real':
        return REAL_LABEL, video_path.stem
    

    parts = video_path.stem.split('_')
    if len(parts) >= 2 and parts[-2].isdigit():
        video_id = '_'.join(parts[-2:])
        return FAKE_LABEL, video_id
    
    return None, None
    


def run_packet_loss_batch(video_info_list, output_h5_path, severity, num_workers):

    print(f"--- Starting Packet Loss Batch for {output_h5_path.name} ---")
    print(f"Processing {len(video_info_list)} videos (Real: {sum(1 for v in video_info_list if v['label'] == REAL_LABEL)}, Fake: {sum(1 for v in video_info_list if v['label'] == FAKE_LABEL)})")
    
    severity_map = {1: 1, 3: 3, 5: 6}
    loss_percentage = severity_map.get(severity, 1)
    all_processed_videos = []
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        h264_input_dir = tmp_path / "h264_input"
        corrupted_output_dir = tmp_path / "corrupted_output"
        h264_input_dir.mkdir()
        corrupted_output_dir.mkdir()
        
        simulation_script_path = PACKET_LOSS_REPO_PATH / "packet_loss_simulation.py"


        print("Step 1/4: Converting videos to H.264 streams (in parallel)...")
        
        workers = os.cpu_count() if num_workers == 0 else num_workers
        tasks = [(video_info['path'], h264_input_dir) for video_info in video_info_list]
        
        with multiprocessing.Pool(processes=workers) as pool:
            list(tqdm(pool.imap_unordered(convert_to_h264, tasks), total=len(tasks), desc="Converting to H.264"))
        

        print(f"Step 2/4: Running Mininet simulation with {loss_percentage}% packet loss...")
        run_cmd = [
            "python3", str(simulation_script_path),
            "--h264_dir", str(h264_input_dir),
            "--dst_path", str(corrupted_output_dir),
            "--switch_loss", str(loss_percentage)
        ]
        result = subprocess.run(run_cmd, text=True)
        if result.returncode != 0:
            print(f"  [Error] Packet loss simulation failed. See stderr below:")
            print(result.stderr)
            return


        print("Step 3/4: Extracting and resizing frames from corrupted videos...")
        result_dir = corrupted_output_dir / str(loss_percentage)
        corrupted_video_paths = list(result_dir.glob("**/*.mp4"))
        

        video_name_to_label = {video_info['path'].stem: video_info['label'] for video_info in video_info_list}
        
        for corrupted_video in tqdm(corrupted_video_paths, desc="Extracting frames"):
            video_name = corrupted_video.stem
            original_label = video_name_to_label.get(video_name)
            
            if original_label is None:
                print(f"Warning: Could not find original label for {video_name}")
                continue
                
            frames = load_video_frames(corrupted_video, NUM_FRAMES_TO_EXTRACT, FRAME_STRIDE)
            if frames:
                resized_frames = resize_frames(frames, TARGET_RESOLUTION)
                all_processed_videos.append({
                    'name': video_name, 
                    'frames': resized_frames,
                    'label': original_label
                })


        print("Step 4/4: Saving results to HDF5 file...")
        if all_processed_videos:
            save_videos_to_hdf5_group(all_processed_videos, output_h5_path)
            print(f"Successfully saved {len(all_processed_videos)} videos to {output_h5_path}")
        else:
            print(f"Warning: No videos were successfully processed for {output_h5_path.name}")



def main():
    print(f"Source Directory: {SOURCE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    

    print("Loading official split info and grouping video paths...")
    video_id_to_split = {}
    for split in ["train", "val", "test"]:
        split_file = SPLIT_DIR / f"{split}.json"
        if not split_file.exists():
            print(f"Warning: Split file not found at {split_file}, skipping this split.")
            continue
        with open(split_file, 'r') as f:
            data = json.load(f)
            for pair in data:
                video_id_to_split[f"{pair[0]}_{pair[1]}"] = split
                video_id_to_split[f"{pair[1]}_{pair[0]}"] = split
                video_id_to_split[pair[0]] = split
                video_id_to_split[pair[1]] = split
    
    video_info_by_split = defaultdict(list)
    all_videos = []
    for ext in VIDEO_EXTENSIONS:
        all_videos.extend(SOURCE_DIR.glob(f'**/*{ext}'))

    for video_path in all_videos:
        if video_path.parent.name == 'corrupt': continue

        label, video_id = get_video_label_and_id(video_path)
        
        if video_id and label is not None:
            split_name = video_id_to_split.get(video_id)
            if split_name:
                video_info_by_split[split_name].append({
                    'path': video_path,
                    'label': label,
                    'id': video_id
                })
    
    for split, video_infos in video_info_by_split.items():
        real_count = sum(1 for v in video_infos if v['label'] == REAL_LABEL)
        fake_count = sum(1 for v in video_infos if v['label'] == FAKE_LABEL)
        print(f"Found {len(video_infos)} videos for '{split}' split (Real: {real_count}, Fake: {fake_count}).")


    for split_name, video_infos in video_info_by_split.items():
        print(f"\n>>> Processing {len(video_infos)} videos for '{split_name}' split... <<<")

        real_videos = [v for v in video_infos if v['label'] == REAL_LABEL]
        fake_videos = [v for v in video_infos if v['label'] == FAKE_LABEL]
        
        print(f"Real videos: {len(real_videos)}, Fake videos: {len(fake_videos)}")

        for severity in SEVERITIES:

            if real_videos:
                real_output_filename = f"{split_name}_packet_loss_real_{severity}.h5"
                real_dst_path = OUTPUT_DIR / real_output_filename
                if real_dst_path.exists():
                    print(f"Skipping, already exists: {real_dst_path}")
                else:
                    print(f"Processing Real videos for {split_name} split, severity {severity}")
                    run_packet_loss_batch(real_videos, real_dst_path, severity, num_workers=10)
            
            
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()
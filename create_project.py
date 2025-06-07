# -*- coding: utf-8 -*-
import os
import sys
import shutil
import subprocess

ALL_FILES = [
    # --- File gốc ---
    (
        "requirements.txt",
        """gdown==5.2.0
PyYAML==6.0.2
opencv-python-headless==4.11.0.86
numpy==2.0.2
ffmpy==0.6.0
psutil==5.9.5
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
"""
    ),
    (
        "setup.py",
        """import os
import subprocess
import sys
import gdown

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER_ID = "16qdCbG0P3cAR-m3P2xZ3q6mKY_QFW-i-"
PRE_TRAINED_MODELS_DIR = os.path.join(SCRIPT_DIR, "pre_trained_models")
REQUIREMENTS_FILE = os.path.join(SCRIPT_DIR, "requirements.txt")

def install_requirements():
    print(">>> Bước 1: Bắt đầu cài đặt các thư viện từ requirements.txt...")
    if not os.path.isfile(REQUIREMENTS_FILE):
        sys.exit(f">>> [Lỗi] Không tìm thấy file requirements.txt tại: {REQUIREMENTS_FILE}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE], check=True, capture_output=True, text=True, encoding='utf-8')
        print(">>> [Thành công] Đã cài đặt xong tất cả các thư viện.")
    except subprocess.CalledProcessError as e:
        sys.exit(f">>> [Lỗi] Cài đặt thất bại:\\n{e.stderr}")

def download_models():
    print(f"\\n>>> Bước 2: Bắt đầu tải models vào '{PRE_TRAINED_MODELS_DIR}'...")
    try:
        gdown.download_folder(id=MODEL_FOLDER_ID, output=PRE_TRAINED_MODELS_DIR, quiet=False, use_cookies=False)
        print(">>> [Thành công] Đã tải xong các model.")
    except Exception as e:
        sys.exit(f">>> [Lỗi] Tải model thất bại: {e}")

def create_project_structure():
    print("\\n>>> Bước 3: Bắt đầu tạo cấu trúc thư mục...")
    dirs_to_create = [
        "data/add_youknow/images", "data/add_youknow/masks",
        "data/mosaic_position/mosaiced_images", "data/mosaic_position/mosaic_masks",
        "data/clean_youknow/original_images", "data/clean_youknow/mosaiced_images", "data/clean_youknow/mosaic_masks",
        "output", "tmp", "pre_trained_models",
        "tool", "script_AI", "script_AI/run", "script_AI/train",
    ]
    for rel_path in dirs_to_create:
        os.makedirs(os.path.join(SCRIPT_DIR, rel_path), exist_ok=True)
    print(">>> [Thành công] Cấu trúc thư mục đã sẵn sàng.")

def main():
    print("="*60); print(" BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG DỰ ÁN UnOrCensored"); print("="*60)
    create_project_structure(); install_requirements(); download_models()
    print("\\n" + "="*60); print(" HOÀN TẤT! Môi trường đã được chuẩn bị."); print("="*60)

if __name__ == "__main__":
    main()"""
    ),
    (
        "run.py",
        """import argparse
import json
import sys
import os
import subprocess
import shutil
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(SCRIPT_DIR, "tool")
SCRIPT_AI_RUN_DIR = os.path.join(SCRIPT_DIR, "script_AI", "run")
TMP_DIR = os.path.join(SCRIPT_DIR, "tmp")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

def run_command(command):
    sys.stderr.write(f"\\n--- EXECUTE: {' '.join(command)} ---\\n")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', cwd=SCRIPT_DIR)
        if process.stderr:
            sys.stderr.write(process.stderr)
        return json.loads(process.stdout)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[Lỗi Script] Lệnh thất bại.\\nLỗi:\\n{e.stderr}")
    except Exception as e:
        sys.exit(f"[Lỗi Script] Lỗi không xác định: {e}")

def handle_image(args, temp_dir):
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.file_path, input_dir)

    common_args = ["--project_root", SCRIPT_DIR]

    if args.task_name == 'add_youknow':
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_add_youknow.py"), "--input_dir", input_dir, "--output_dir", output_dir] + common_args)
    elif args.task_name == 'clean_youknow':
        mask_dir = os.path.join(temp_dir, "generated_masks")
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_mosaic_position.py"), "--input_dir", input_dir, "--output_dir", mask_dir] + common_args)
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_clean_youknow.py"), "--input_dir", input_dir, "--mask_dir", mask_dir, "--output_dir", output_dir] + common_args)
    
    final_dest_dir = args.folder_path or OUTPUT_DIR
    os.makedirs(final_dest_dir, exist_ok=True)
    final_dest_file = os.path.join(final_dest_dir, os.path.basename(args.file_path))
    shutil.move(os.path.join(output_dir, os.path.basename(args.file_path)), final_dest_file)
    sys.stderr.write(f"Đã lưu kết quả vào: {final_dest_file}\\n")

def handle_video(args, temp_dir):
    common_args = ["--project_root", SCRIPT_DIR]
    info = run_command(["python", os.path.join(TOOL_DIR, "get_file_info.py"), "--input", args.file_path])
    metadata_json = json.dumps(info.get("metadata", {}))
    duration = run_command(["python", os.path.join(TOOL_DIR, "duration_split.py"), "--input", args.file_path, "--fps", str(info['metadata']['fps'])]).get("optimal_chunk_duration", 300)
    
    chunks_dir = os.path.join(temp_dir, "chunks")
    chunk_paths = run_command(["python", os.path.join(TOOL_DIR, "split_video.py"), "--input", args.file_path, "--duration", str(duration), "--output_dir", chunks_dir]).get("chunk_paths", [])
    
    processed_chunks = []
    for i, chunk_path in enumerate(chunk_paths):
        chunk_proc_dir = os.path.join(temp_dir, f"chunk_{i}")
        frames_info = run_command(["python", os.path.join(TOOL_DIR, "video_to_frames.py"), "--input", chunk_path, "--output_dir", chunk_proc_dir])
        
        if args.task_name == 'add_youknow':
            processed_frames_dir = os.path.join(chunk_proc_dir, "processed")
            run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_add_youknow.py"), "--input_dir", frames_info['frame_folder'], "--output_dir", processed_frames_dir] + common_args)
        elif args.task_name == 'clean_youknow':
            mask_dir = os.path.join(chunk_proc_dir, "masks")
            processed_frames_dir = os.path.join(chunk_proc_dir, "processed")
            run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_mosaic_position.py"), "--input_dir", frames_info['frame_folder'], "--output_dir", mask_dir] + common_args)
            run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_clean_youknow.py"), "--input_dir", frames_info['frame_folder'], "--mask_dir", mask_dir, "--output_dir", processed_frames_dir] + common_args)

        proc_chunk_cmd = ["python", os.path.join(TOOL_DIR, "frames_to_video.py"), "--frame_folder", processed_frames_dir, "--metadata_json", metadata_json, "--output", os.path.join(chunk_proc_dir, "out.mp4")]
        if frames_info.get('audio_path'): proc_chunk_cmd.extend(["--audio_path", frames_info['audio_path']])
        processed_chunks.append(run_command(proc_chunk_cmd).get("output_path"))

    list_file = os.path.join(temp_dir, "mergelist.txt")
    with open(list_file, "w", encoding='utf-8') as f:
        for p in processed_chunks:
            # === SỬA LỖI TẠI ĐÂY ===
            clean_path = p.replace('\\\\', '/')
            f.write(f"file '{clean_path}'\\n")
        
    final_path = os.path.join(args.folder_path or OUTPUT_DIR, f"final_{os.path.basename(args.file_path)}")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    run_command(["python", os.path.join(TOOL_DIR, "merge_video.py"), "--input_list_file", list_file, "--output", final_path])

def main():
    parser = argparse.ArgumentParser(description="Script điều phối chính cho dự án UnOrCensored.")
    parser.add_argument('--file_path', required=True)
    parser.add_argument('--task_name', required=True, choices=['add_youknow', 'clean_youknow'])
    parser.add_argument('--folder_path', help="Thư mục đầu ra (tùy chọn).")
    args = parser.parse_args()
    temp_dir = os.path.join(TMP_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(temp_dir, exist_ok=True)
    try:
        file_type = run_command(["python", os.path.join(TOOL_DIR, "get_file_type.py"), "--input", args.file_path]).get("file_type")
        if file_type == 'image': handle_image(args, temp_dir)
        elif file_type == 'video': handle_video(args, temp_dir)
    finally:
        shutil.rmtree(temp_dir); sys.stderr.write("\\nHoàn tất và đã dọn dẹp file tạm.\\n")
if __name__ == "__main__": main()"""
    ),
    (
        "train.py",
        """import argparse, sys, os, datetime
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR) # Đảm bảo có thể import script_AI
from script_AI.model_loader import load_model
from script_AI.train import train_segmentation, train_clean_youknow
def main():
    parser = argparse.ArgumentParser(description="Script điều phối chính cho việc tinh chỉnh model.")
    parser.add_argument('--pre_trained_models_name', required=True, choices=['add_youknow', 'mosaic_position', 'clean_youknow'])
    parser.add_argument('--folder_path', help="Đường dẫn thư mục dữ liệu. Mặc định: 'data/<model_name>'.")
    parser.add_argument('--epochs', type=int, default=10); parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    data_path = args.folder_path or os.path.join(SCRIPT_DIR, 'data', args.pre_trained_models_name)
    if not os.path.isdir(data_path): sys.exit(f"[Lỗi] Thư mục dữ liệu không tồn tại: {data_path}")
    base_model_path = os.path.join(SCRIPT_DIR, 'pre_trained_models', f"{args.pre_trained_models_name}.pth")
    model = load_model(args.pre_trained_models_name, base_model_path)
    save_path = os.path.join(SCRIPT_DIR, 'pre_trained_models', f"{args.pre_trained_models_name}_finetuned_{datetime.datetime.now().strftime('%Y%m%d')}.pth")
    try:
        if args.pre_trained_models_name in ['add_youknow', 'mosaic_position']:
            train_segmentation.run_training_session(model, data_path, args.epochs, args.lr, save_path)
        elif args.pre_trained_models_name == 'clean_youknow':
            train_clean_youknow.run_training_session_inpaint(model, data_path, args.epochs, args.lr, save_path)
    except Exception as e: sys.exit(f"\\n[Lỗi] Huấn luyện thất bại: {e}")
    sys.stderr.write(f"\\nHoàn tất! Model đã tinh chỉnh được lưu tại: {save_path}\\n")
if __name__ == '__main__': main()"""
    ),
    # --- tool/ ---
    ("tool/get_file_type.py", """import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    video_ext = ['.mp4', '.avi', '.mov', '.mkv']
    image_ext = ['.jpg', '.jpeg', '.png', '.bmp']
    ext = os.path.splitext(args.input)[1].lower()
    if ext in video_ext:
        print(json.dumps({"file_type": "video"}))
    elif ext in image_ext:
        print(json.dumps({"file_type": "image"}))
    else:
        print(json.dumps({"file_type": "unknown"}))

if __name__ == "__main__":
    main()"""),
    ("tool/get_file_info.py", """import cv2
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(json.dumps({"error": f"Không thể mở video: {args.input}"}))
        return
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        print(json.dumps({'metadata': {'fps': fps, 'width': width, 'height': height, 'frame_count': frame_count, 'duration_seconds': duration}}))
    finally:
        cap.release()

if __name__ == "__main__":
    main()"""),
    ("tool/cut_video.py", """import argparse, json, sys, os
from ffmpy import FFmpeg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', default=None)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    opts = ['-c', 'copy']
    if args.end:
        opts.extend(['-to', args.end])
    ff = FFmpeg(global_options=['-y'], inputs={args.input: ['-ss', args.start]}, outputs={args.output: opts})
    ff.run(stdout=sys.stderr, stderr=sys.stderr)
    print(json.dumps({'output_path': os.path.abspath(args.output)}))

if __name__ == "__main__":
    main()"""),
    ("tool/duration_split.py", """import argparse, json, sys, os, psutil
from ffmpy import FFmpeg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--fps', type=float, required=True)
    args = parser.parse_args()
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_frame = os.path.join(temp_dir, f"temp_frame.png")
    ff = FFmpeg(global_options=['-y'], inputs={args.input: ['-ss', '00:00:01']}, outputs={temp_frame: ['-vframes', '1']})
    duration = 300
    try:
        ff.run(stderr=sys.stderr)
        data_rate = os.path.getsize(temp_frame) * 3 * args.fps
        usable_space = psutil.disk_usage(os.path.abspath('.')).free * 0.8
        duration = int(usable_space / data_rate) if data_rate > 0 else 300
    finally:
        if os.path.exists(temp_frame):
            os.remove(temp_frame)
        print(json.dumps({'optimal_chunk_duration': duration}))

if __name__ == "__main__":
    main()"""),
    ("tool/split_video.py", """import argparse, json, sys, os
from ffmpy import FFmpeg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--duration', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pattern = os.path.join(args.output_dir, f'chunk_%05d{os.path.splitext(args.input)[1]}')
    ff = FFmpeg(global_options=['-y'], inputs={args.input: None}, outputs={pattern: ['-c', 'copy', '-map', '0', '-segment_time', str(args.duration), '-f', 'segment', '-reset_timestamps', '1']})
    ff.run(stderr=sys.stderr)
    chunks = sorted([os.path.abspath(os.path.join(args.output_dir, f)) for f in os.listdir(args.output_dir) if f.startswith('chunk_')])
    print(json.dumps({'chunk_paths': chunks}))

if __name__ == "__main__":
    main()"""),
    ("tool/video_to_frames.py", """import argparse, json, sys, os
from ffmpy import FFmpeg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    frame_dir = os.path.join(args.output_dir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    audio_path = os.path.join(args.output_dir, 'audio.aac')
    frame_pattern = os.path.join(frame_dir, '%08d.png')
    ff_frames = FFmpeg(global_options=['-y'], inputs={args.input: None}, outputs={frame_pattern: ['-qscale:v', '2']})
    ff_audio = FFmpeg(global_options=['-y'], inputs={args.input: None}, outputs={audio_path: ['-vn', '-acodec', 'copy']})
    ff_frames.run(stderr=sys.stderr)
    aop = None
    try:
        ff_audio.run(stderr=sys.stderr)
        aop = os.path.abspath(audio_path)
    except:
        pass
    print(json.dumps({'frame_folder': os.path.abspath(frame_dir), 'audio_path': aop}))

if __name__ == "__main__":
    main()"""),
    ("tool/frames_to_video.py", """import argparse, json, sys, os
from ffmpy import FFmpeg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_folder', required=True)
    parser.add_argument('--audio_path', default=None)
    parser.add_argument('--metadata_json', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fps = json.loads(args.metadata_json).get('fps')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pattern = os.path.join(args.frame_folder, '%08d.png')
    inputs = {pattern: ['-framerate', str(fps)]}
    if args.audio_path and os.path.exists(args.audio_path):
        inputs[args.audio_path] = None
    opts = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
    if args.audio_path and os.path.exists(args.audio_path):
        opts.extend(['-c:a', 'aac', '-shortest'])
    ff = FFmpeg(global_options=['-y'], inputs=inputs, outputs={args.output: opts})
    ff.run(stderr=sys.stderr)
    print(json.dumps({'output_path': os.path.abspath(args.output)}))

if __name__ == "__main__":
    main()"""),
    ("tool/merge_video.py", """import argparse, json, sys, os
from ffmpy import FFmpeg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_list_file', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ff = FFmpeg(global_options=['-y'], inputs={args.input_list_file: ['-f', 'concat', '-safe', '0']}, outputs={args.output: ['-c', 'copy']})
    ff.run(stderr=sys.stderr)
    print(json.dumps({'final_video_path': os.path.abspath(args.output)}))

if __name__ == "__main__":
    main()"""),
    # --- script_AI (Sửa lỗi và làm cho dễ đọc hơn) ---
    ("script_AI/__init__.py", ""),
    ("script_AI/train/__init__.py", ""),
    (
        "script_AI/images_processing.py",
        """import torch, cv2, numpy as np
from torchvision import transforms

def load_image_to_tensor(path, size=(512,512), normalize=True):
    img = cv2.imread(path)
    original_size = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform_list = [transforms.ToTensor(), transforms.Resize(size, antialias=True)]
    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    transform = transforms.Compose(transform_list)
    return transform(img).unsqueeze(0), original_size

def save_tensor_as_image(tensor, out_path, original_size, is_inpaint=False):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = transforms.Resize(original_size, antialias=True)(tensor)
    if is_inpaint:
        tensor = (tensor * 0.5) + 0.5
    img_np = np.clip(tensor.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

def load_mask_to_tensor(path, size=(512,512)):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return transforms.Compose([transforms.ToTensor(), transforms.Resize(size, antialias=True)])(mask).unsqueeze(0)"""
    ),
    (
        "script_AI/model_loader.py",
        """import torch, os, sys
# Thêm thư mục gốc vào path để đảm bảo import hoạt động
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script_AI.models import AddYouknowModel, CleanYouknowModel

def load_model(name, pth, dev='cpu'):
    if name in ['add_youknow', 'mosaic_position']:
        model = AddYouknowModel()
    elif name == 'clean_youknow':
        model = CleanYouknowModel()
    else:
        raise ValueError(f"Model không hỗ trợ: {name}")
    state_dict = torch.load(pth, map_location=torch.device(dev))
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict({k.replace('module.',''): v for k, v in state_dict.items()})
    return model.to(dev).eval()"""
    ),
    (
        "script_AI/models.py",
        """import torch, torch.nn as nn, torch.nn.functional as F

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0','resnet18', pretrained=True, verbose=False)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
    def forward(self, x):
        return self.features(x)

class AddYouknowModel(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()
        self.backbone = ResNet18()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, n_class, 3, padding=1)
        )
    def forward(self, x):
        return F.interpolate(self.decoder(self.backbone(x)), size=x.shape[2:], mode='bilinear', align_corners=False)

class PConv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none', activ='relu', bias=False):
        super().__init__()
        self.sampler = None
        if sample == 'down': self.sampler = nn.MaxPool2d(2)
        elif sample == 'up': self.sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias)
        self.mask_conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters(): p.requires_grad = False
        
        self.bn = nn.BatchNorm2d(out_ch) if bn else None
        self.activ = nn.ReLU(True) if activ == 'relu' else nn.LeakyReLU(0.2, True)

    def forward(self, x, mask):
        if self.sampler: x, mask = self.sampler(x), self.sampler(mask)
        with torch.no_grad():
            mask_ratio = 1 / (self.mask_conv(mask) + 1e-8)
        output = self.conv(x * mask) * mask_ratio
        if self.bn: output = self.bn(output)
        return self.activ(output), mask

class CleanYouknowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1=PConv(4,64,bn=False,sample='down'); self.e2=PConv(64,128,sample='down')
        self.e3=PConv(128,256,sample='down'); self.e4=PConv(256,512,sample='down')
        self.d1=PConv(512+256,256,activ='leaky',sample='up'); self.d2=PConv(256+128,128,activ='leaky',sample='up')
        self.d3=PConv(128+64,64,activ='leaky',sample='up'); self.d4=PConv(64+4,3,bn=False,activ='leaky')
        self.final=nn.Conv2d(3,3,1)

    def forward(self, x, mask):
        e1,m1=self.e1(x,mask); e2,m2=self.e2(e1,m1); e3,m3=self.e3(e2,m2); e4,m4=self.e4(e3,m3)
        d1,_=self.d1(torch.cat([e4,e3],1),torch.cat([m4,m3],1)); d2,_=self.d2(torch.cat([d1,e2],1),torch.cat([m2,m2],1))
        d3,_=self.d3(torch.cat([d2,e1],1),torch.cat([m2,m1],1)); d4,_=self.d4(torch.cat([d3,x],1),torch.cat([m1,mask],1))
        return torch.tanh(self.final(d4))"""
    ),
    (
        "script_AI/run/run_add_youknow.py",
        """import argparse,json,sys,os,torch;from tqdm import tqdm;
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))));
from script_AI.model_loader import load_model; from script_AI.images_processing import load_image_to_tensor,save_tensor_as_image
def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--project_root',required=True);a=p.parse_args();
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(a.project_root, 'pre_trained_models/add_youknow.pth')
    model=load_model('add_youknow', model_path, dev)
    os.makedirs(a.output_dir,exist_ok=True)
    for fn in tqdm(os.listdir(a.input_dir),file=sys.stderr):
        try:
            t,s=load_image_to_tensor(os.path.join(a.input_dir,fn));
            with torch.no_grad(): mt=torch.sigmoid(model(t.to(dev)))
            save_tensor_as_image(mt,os.path.join(a.output_dir,fn),s)
        except Exception as e: sys.stderr.write(f"Lỗi file {fn}: {e}\\n")
    print(json.dumps({"processed_frame_folder":os.path.abspath(a.output_dir)}))
if __name__=="__main__":main()"""
    ),
    (
        "script_AI/run/run_mosaic_position.py",
        """import argparse,json,sys,os,torch;from tqdm import tqdm;
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))));
from script_AI.model_loader import load_model; from script_AI.images_processing import load_image_to_tensor,save_tensor_as_image
def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--project_root',required=True);a=p.parse_args();
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(a.project_root, 'pre_trained_models/mosaic_position.pth')
    model=load_model('mosaic_position', model_path, dev)
    os.makedirs(a.output_dir,exist_ok=True)
    for fn in tqdm(os.listdir(a.input_dir),file=sys.stderr):
        try:
            t,s=load_image_to_tensor(os.path.join(a.input_dir,fn));
            with torch.no_grad(): mt=torch.sigmoid(model(t.to(dev)))
            save_tensor_as_image(mt,os.path.join(a.output_dir,fn),s)
        except Exception as e: sys.stderr.write(f"Lỗi file {fn}: {e}\\n")
    print(json.dumps({"processed_frame_folder":os.path.abspath(a.output_dir)}))
if __name__=="__main__":main()"""
    ),
    (
        "script_AI/run/run_clean_youknow.py",
        """import argparse,json,sys,os,torch;from tqdm import tqdm;
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))));
from script_AI.model_loader import load_model; from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor,save_tensor_as_image
def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--mask_dir',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--project_root',required=True);a=p.parse_args();
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(a.project_root, 'pre_trained_models/clean_youknow.pth')
    model=load_model('clean_youknow', model_path, dev)
    os.makedirs(a.output_dir,exist_ok=True)
    for fn in tqdm(os.listdir(a.input_dir),file=sys.stderr):
        mp=os.path.join(a.mask_dir,fn);
        if not os.path.exists(mp):continue
        try:
            it,s=load_image_to_tensor(os.path.join(a.input_dir,fn),norm=False);it_norm=it*2-1
            mt=load_mask_to_tensor(mp);it_norm,mt=it_norm.to(dev),mt.to(dev)
            with torch.no_grad():
                out=model(torch.cat((it_norm,mt),dim=1),mt)
                final=(out*mt)+(it_norm*(1-mt))
            save_tensor_as_image(final,os.path.join(a.output_dir,fn),s,is_inpaint=True)
        except Exception as e:sys.stderr.write(f"Lỗi file {fn}: {e}\\n")
    print(json.dumps({"processed_frame_folder":os.path.abspath(a.output_dir)}))
if __name__=="__main__":main()"""
    ),
    ("script_AI/train/train_segmentation.py", "import torch,os,sys;import torch.nn as nn;import torch.optim as optim;from torch.utils.data import Dataset,DataLoader;from tqdm import tqdm;sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor;class S(Dataset):def __init__(s,i,m):s.i,s.m,s.f=i,m,sorted(os.listdir(i));def __len__(s):return len(s.f);def __getitem__(s,i):im,_=load_image_to_tensor(os.path.join(s.i,s.f[i]));ms=load_mask_to_tensor(os.path.join(s.m,s.f[i]));return im.squeeze(0),ms.squeeze(0);def t(m,d,l,o,c):m.train();[o.step()for i,k in tqdm(l,file=sys.stderr)for i,k in[(i.to(d),k.to(d))]for _ in[o.zero_grad()]for loss in[c(m(i),k)]for _ in[loss.backward()]];def r(m,p,e,lr,s):d='cuda'if torch.cuda.is_available()else'cpu';m.to(d);ds=S(os.path.join(p,'images'),os.path.join(p,'masks'));l=DataLoader(ds,batch_size=4,shuffle=True,num_workers=2);op=optim.Adam(m.parameters(),lr=lr);cr=nn.BCEWithLogitsLoss();[t(m,d,l,op,cr)or torch.save(m.state_dict(),s)for ep in range(e)]"),
    ("script_AI/train/train_clean_youknow.py", "import torch,os,sys;import torch.nn as nn;import torch.optim as optim;from torch.utils.data import Dataset,DataLoader;from tqdm import tqdm;sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor;class I(Dataset):def __init__(s,p):s.g,s.m,s.k=os.path.join(p,'original_images'),os.path.join(p,'mosaiced_images'),os.path.join(p,'mosaic_masks');s.f=sorted(os.listdir(s.g));def __len__(s):return len(s.f);def __getitem__(s,i):gt,_=load_image_to_tensor(os.path.join(s.g,s.f[i]),norm=False);mo,_=load_image_to_tensor(os.path.join(s.m,s.f[i]),norm=False);msk=load_mask_to_tensor(os.path.join(s.k,s.f[i]));return gt.squeeze(0),mo.squeeze(0),msk.squeeze(0);def t(m,d,l,o,c):m.train();[o.step()for gt,mo,msk in tqdm(l,file=sys.stderr)for gt,mo,msk in[(gt.to(d),mo.to(d),msk.to(d))]for mi in[torch.cat((mo*2-1,msk),1)]for _ in[o.zero_grad()]for out in[m(mi,msk)]for loss in[c(out*msk,gt*msk)]for _ in[loss.backward()]];def r(m,p,e,lr,s):d='cuda'if torch.cuda.is_available()else'cpu';m.to(d);ds=I(p);l=DataLoader(ds,batch_size=2,shuffle=True,num_workers=2);op=optim.Adam(m.parameters(),lr=lr);cr=nn.L1Loss();[t(m,d,l,op,cr)or torch.save(m.state_dict(),s)for ep in range(e)]"),
    ("README.md", "Dự án UnOrCensored. Cài đặt: `python setup.py`. Sử dụng: `python run.py --file_path <p> --task_name <t>`"),
]

def create_project_files(base_dir="UnOrCensored_Project"):
    if os.path.exists(base_dir):
        print(f"Thư mục '{base_dir}' đã tồn tại. Vui lòng xóa hoặc đổi tên và chạy lại.")
        return
    os.makedirs(base_dir)
    print(f"Đang tạo dự án trong thư mục: {os.path.abspath(base_dir)}")
    for file_path, file_content in ALL_FILES:
        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(file_content.strip())
        print(f"Đã tạo file: {full_path}")
    print(f"\\n[Thành công] Đã tạo xong dự án. Vui lòng chạy 'python {os.path.join(base_dir, 'setup.py')}' để cài đặt.")

if __name__ == "__main__":
    create_project_files()



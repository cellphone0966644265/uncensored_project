# -*- coding: utf-8 -*-
import os
import sys
import shutil

# Danh sách chứa toàn bộ các file của dự án
# Mỗi phần tử là một tuple: (đường_dẫn_file, nội_dung_file)
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
        """# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import gdown

# --- Lấy đường dẫn tuyệt đối của thư mục chứa script này ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Cấu hình ---
MODEL_FOLDER_ID = "16qdCbG0P3cAR-m3P2xZ3q6mKY_QFW-i-"
PRE_TRAINED_MODELS_DIR = os.path.join(SCRIPT_DIR, "pre_trained_models")
REQUIREMENTS_FILE = os.path.join(SCRIPT_DIR, "requirements.txt")

def install_requirements():
    """
    Cài đặt tất cả thư viện từ requirements.txt.
    Đây là phương pháp cài đặt gốc, phù hợp với môi trường Colab.
    """
    print(">>> Bước 1: Bắt đầu cài đặt các thư viện từ requirements.txt...")
    if not os.path.isfile(REQUIREMENTS_FILE):
        print(f">>> [Lỗi] Không tìm thấy file requirements.txt tại: {REQUIREMENTS_FILE}", file=sys.stderr)
        sys.exit(1)
    try:
        # Chạy một lệnh pip install duy nhất
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE], check=True, capture_output=True, text=True, encoding='utf-8')
        print(">>> [Thành công] Đã cài đặt xong tất cả các thư viện.")
    except subprocess.CalledProcessError as e:
        print(">>> [Lỗi] Cài đặt thất bại. Vui lòng kiểm tra log bên dưới.", file=sys.stderr)
        print(f"Lỗi chi tiết:\\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def download_models():
    """Tải các model đã được huấn luyện."""
    print(f"\\n>>> Bước 2: Bắt đầu tải models vào '{PRE_TRAINED_MODELS_DIR}'...")
    try:
        gdown.download_folder(id=MODEL_FOLDER_ID, output=PRE_TRAINED_MODELS_DIR, quiet=False, use_cookies=False)
        print(">>> [Thành công] Đã tải xong các model.")
    except Exception as e:
        print(f">>> [Lỗi] Tải model thất bại: {e}", file=sys.stderr)
        sys.exit(1)

def create_project_structure():
    """Tạo cấu trúc thư mục cần thiết."""
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
    print("="*60)
    print(" BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG DỰ ÁN UnOrCensored")
    print("="*60)
    create_project_structure()
    install_requirements()
    download_models()
    print("\\n" + "="*60)
    print(" HOÀN TẤT! Môi trường đã được chuẩn bị.")
    print("="*60)

if __name__ == "__main__":
    main()"""
    ),
    (
        "run.py",
        """# -*- coding: utf-8 -*-
import argparse, json, sys, os, subprocess, shutil, datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(SCRIPT_DIR, "tool")
SCRIPT_AI_RUN_DIR = os.path.join(SCRIPT_DIR, "script_AI", "run")
TMP_DIR = os.path.join(SCRIPT_DIR, "tmp")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

def run_command(command):
    sys.stderr.write(f"\\n--- EXECUTE: {' '.join(command)} ---\\n")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        if process.stderr: sys.stderr.write(process.stderr)
        result = json.loads(process.stdout)
        if "error" in result:
            sys.stderr.write(f"[Lỗi Module] {result.get('error')}\\n")
            sys.exit(1)
        return result
    except Exception as e:
        sys.stderr.write(f"[Lỗi Script] Lệnh thất bại: {e}\\n")
        sys.exit(1)

def handle_image(args, temp_dir):
    input_dir = os.path.join(temp_dir, "input")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.file_path, input_dir)

    # Chỉ phát hiện và xuất ra mask
    if args.task_name == 'add_youknow':
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_add_youknow.py"), "--input_dir", input_dir, "--output_dir", output_dir])
    
    # Quy trình 2 bước: phát hiện vị trí -> tái tạo
    elif args.task_name == 'clean_youknow':
        sys.stderr.write("Bắt đầu quy trình 2 bước cho 'clean_youknow' trên ảnh...\\n")
        mask_dir = os.path.join(temp_dir, "generated_masks")
        
        sys.stderr.write("Bước 1: Chạy 'mosaic_position' để tạo mask...\\n")
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_mosaic_position.py"), "--input_dir", input_dir, "--output_dir", mask_dir])

        sys.stderr.write("Bước 2: Chạy 'clean_youknow' để tái tạo ảnh...\\n")
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_clean_youknow.py"), "--input_dir", input_dir, "--mask_dir", mask_dir, "--output_dir", output_dir])
    
    final_output_path = os.path.join(output_dir, os.path.basename(args.file_path))
    final_dest_dir = args.folder_path or OUTPUT_DIR
    os.makedirs(final_dest_dir, exist_ok=True)
    final_dest_file = os.path.join(final_dest_dir, os.path.basename(args.file_path))
    shutil.move(final_output_path, final_dest_file)
    sys.stderr.write(f"Đã lưu kết quả vào: {final_dest_file}\\n")

def handle_video(args, temp_dir):
    info = run_command(["python", os.path.join(TOOL_DIR, "get_file_info.py"), "--input", args.file_path])
    metadata_json = json.dumps(info.get("metadata", {}))
    duration = run_command(["python", os.path.join(TOOL_DIR, "duration_split.py"), "--input", args.file_path, "--fps", str(info['metadata']['fps'])]).get("optimal_chunk_duration", 300)
    
    chunks_dir = os.path.join(temp_dir, "chunks")
    chunk_paths = run_command(["python", os.path.join(TOOL_DIR, "split_video.py"), "--input", args.file_path, "--duration", str(duration), "--output_dir", chunks_dir]).get("chunk_paths", [])
    
    processed_chunks = []
    for i, chunk_path in enumerate(chunk_paths):
        sys.stderr.write(f"\\n>>> Đang xử lý chunk {i+1}/{len(chunk_paths)}...\\n")
        chunk_proc_dir = os.path.join(temp_dir, f"chunk_{i}")
        
        frames_info = run_command(["python", os.path.join(TOOL_DIR, "video_to_frames.py"), "--input", chunk_path, "--output_dir", chunk_proc_dir])
        frames_folder = frames_info['frame_folder']
        
        if args.task_name == 'add_youknow':
            processed_frames_dir = os.path.join(chunk_proc_dir, "processed_frames")
            run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_add_youknow.py"), "--input_dir", frames_folder, "--output_dir", processed_frames_dir])
        elif args.task_name == 'clean_youknow':
            mask_dir = os.path.join(chunk_proc_dir, "generated_masks")
            processed_frames_dir = os.path.join(chunk_proc_dir, "processed_frames")
            sys.stderr.write("Bước A: Chạy 'mosaic_position' để tạo masks cho các frame...\\n")
            run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_mosaic_position.py"), "--input_dir", frames_folder, "--output_dir", mask_dir])
            sys.stderr.write("Bước B: Chạy 'clean_youknow' để tái tạo các frame...\\n")
            run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_clean_youknow.py"), "--input_dir", frames_folder, "--mask_dir", mask_dir, "--output_dir", processed_frames_dir])

        proc_chunk_cmd = ["python", os.path.join(TOOL_DIR, "frames_to_video.py"), "--frame_folder", processed_frames_dir, "--metadata_json", metadata_json, "--output", os.path.join(chunk_proc_dir, "out.mp4")]
        if frames_info.get('audio_path'): proc_chunk_cmd.extend(["--audio_path", frames_info['audio_path']])
        processed_chunks.append(run_command(proc_chunk_cmd).get("output_path"))

    list_file = os.path.join(temp_dir, "mergelist.txt")
    with open(list_file, "w", encoding='utf-8') as f:
        for p in processed_chunks: f.write(f"file '{p.replace('\\\\', '/')}'\\n")
        
    final_path = os.path.join(args.folder_path or OUTPUT_DIR, f"final_{os.path.basename(args.file_path)}")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    run_command(["python", os.path.join(TOOL_DIR, "merge_video.py"), "--input_list_file", list_file, "--output", final_path])

def main():
    parser = argparse.ArgumentParser(description="Script điều phối chính cho dự án UnOrCensored.")
    parser.add_argument('--file_path', required=True, help="Đường dẫn file đầu vào.")
    parser.add_argument('--task_name', required=True, choices=['add_youknow', 'clean_youknow'], help="Tên tác vụ: 'add_youknow' (chỉ tạo mask) hoặc 'clean_youknow' (tạo mask rồi tái tạo).")
    parser.add_argument('--folder_path', help="Thư mục đầu ra (tùy chọn).")
    args = parser.parse_args()

    temp_dir = os.path.join(TMP_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(temp_dir, exist_ok=True)
    try:
        file_type = run_command(["python", os.path.join(TOOL_DIR, "get_file_type.py"), "--input", args.file_path]).get("file_type")
        if file_type == 'image': handle_image(args, temp_dir)
        elif file_type == 'video': handle_video(args, temp_dir)
    finally:
        shutil.rmtree(temp_dir)
        sys.stderr.write("\\nHoàn tất và đã dọn dẹp file tạm.\\n")

if __name__ == "__main__":
    main()"""
    ),
    (
        "train.py",
        """# -*- coding: utf-8 -*-
import argparse, sys, os, datetime
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from script_AI.model_loader import load_model
from script_AI.train import train_segmentation, train_clean_youknow

def main():
    parser = argparse.ArgumentParser(description="Script điều phối chính cho việc tinh chỉnh model.")
    parser.add_argument('--pre_trained_models_name', required=True, choices=['add_youknow', 'mosaic_position', 'clean_youknow'], help="Tên model cần tinh chỉnh.")
    parser.add_argument('--folder_path', help="Đường dẫn thư mục dữ liệu. Mặc định: 'data/<model_name>'.")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    data_path = args.folder_path or os.path.join(SCRIPT_DIR, 'data', args.pre_trained_models_name)
    if not os.path.isdir(data_path):
        sys.exit(f"[Lỗi] Thư mục dữ liệu không tồn tại: {data_path}")

    base_model_path = os.path.join(SCRIPT_DIR, 'pre_trained_models', f"{args.pre_trained_models_name}.pth")
    model = load_model(args.pre_trained_models_name, base_model_path)
    save_path = os.path.join(SCRIPT_DIR, 'pre_trained_models', f"{args.pre_trained_models_name}_finetuned_{datetime.datetime.now().strftime('%Y%m%d')}.pth")

    try:
        if args.pre_trained_models_name in ['add_youknow', 'mosaic_position']:
            train_segmentation.run_training_session(model, data_path, args.epochs, args.lr, save_path)
        elif args.pre_trained_models_name == 'clean_youknow':
            train_clean_youknow.run_training_session_inpaint(model, data_path, args.epochs, args.lr, save_path)
    except Exception as e:
        sys.exit(f"\\n[Lỗi] Huấn luyện thất bại: {e}")
        
    sys.stderr.write(f"\\nHoàn tất! Model đã tinh chỉnh được lưu tại: {save_path}\\n")

if __name__ == '__main__':
    main()"""
    ),
    # --- tool/ ---
    ("tool/get_file_type.py", """# -*- coding: utf-8 -*-
import os, argparse, json, sys
def get_file_type(file_path):
    video_ext = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']; image_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    ext = os.path.splitext(file_path)[1].lower()
    if ext in video_ext: return {"file_type": "video"}
    if ext in image_ext: return {"file_type": "image"}
    return {"file_type": "unknown"}
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', required=True); args = parser.parse_args()
    print(json.dumps(get_file_type(args.input)))
if __name__ == "__main__": main()"""),
    ("tool/get_file_info.py", """# -*- coding: utf-8 -*-
import cv2, argparse, json, sys
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return {"error": f"Không thể mở: {video_path}"}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return {"metadata": {"fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0}}
    finally: cap.release()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', required=True); args = parser.parse_args()
    print(json.dumps(get_video_info(args.input)))
if __name__ == "__main__": main()"""),
    ("tool/cut_video.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os; from ffmpy import FFmpeg
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', required=True); parser.add_argument('--output', required=True); parser.add_argument('--start', required=True); parser.add_argument('--end', default=None); args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    opts = ['-c', 'copy'];
    if args.end: opts.extend(['-to', args.end])
    ff = FFmpeg(global_options=['-y'], inputs={args.input: ['-ss', args.start]}, outputs={args.output: opts})
    try:
        ff.run(stdout=sys.stderr, stderr=sys.stderr)
        print(json.dumps({"output_path": os.path.abspath(args.output)}))
    except Exception as e: print(json.dumps({"error": "Lỗi FFmpeg", "details": str(e)}))
if __name__ == "__main__": main()"""),
    ("tool/duration_split.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os,psutil; from ffmpy import FFmpeg
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', required=True); parser.add_argument('--fps', type=float, required=True); args = parser.parse_args()
    temp_dir = "tmp"; os.makedirs(temp_dir, exist_ok=True)
    temp_frame = os.path.join(temp_dir, f"temp_frame.png")
    ff = FFmpeg(global_options=['-y'], inputs={args.input:['-ss','00:00:01']}, outputs={temp_frame:['-vframes','1']})
    try:
        ff.run(stderr=sys.stderr)
        data_rate = os.path.getsize(temp_frame) * 3 * args.fps
        usable_space = psutil.disk_usage('.').free * 0.8
        duration = int(usable_space / data_rate) if data_rate > 0 else 300
        print(json.dumps({"optimal_chunk_duration": duration}))
    except: print(json.dumps({"optimal_chunk_duration": 300}))
    finally:
        if os.path.exists(temp_frame): os.remove(temp_frame)
if __name__ == "__main__": main()"""),
    ("tool/split_video.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os; from ffmpy import FFmpeg
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', required=True); parser.add_argument('--duration', type=int, required=True); parser.add_argument('--output_dir', required=True); args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pattern = os.path.join(args.output_dir, f"chunk_%05d{os.path.splitext(args.input)[1]}")
    ff = FFmpeg(global_options=['-y'], inputs={args.input:None}, outputs={pattern:['-c','copy','-map','0','-segment_time',str(args.duration),'-f','segment','-reset_timestamps','1']})
    try:
        ff.run(stderr=sys.stderr)
        chunks = sorted([os.path.abspath(os.path.join(args.output_dir,f)) for f in os.listdir(args.output_dir) if f.startswith('chunk_')])
        print(json.dumps({"chunk_paths": chunks}))
    except Exception as e: print(json.dumps({"error":"Lỗi FFmpeg", "details":str(e)}))
if __name__ == "__main__": main()"""),
    ("tool/video_to_frames.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os; from ffmpy import FFmpeg
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input', required=True); parser.add_argument('--output_dir', required=True); args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True); frame_dir = os.path.join(args.output_dir, 'frames'); os.makedirs(frame_dir, exist_ok=True)
    audio_path = os.path.join(args.output_dir, "audio.aac"); frame_pattern = os.path.join(frame_dir, '%08d.png')
    ff_frames = FFmpeg(global_options=['-y'], inputs={args.input: None}, outputs={frame_pattern: ['-qscale:v', '2']})
    ff_audio = FFmpeg(global_options=['-y'], inputs={args.input: None}, outputs={audio_path: ['-vn', '-acodec', 'copy']})
    try:
        ff_frames.run(stderr=sys.stderr); audio_out_path = None
        try: ff_audio.run(stderr=sys.stderr); audio_out_path = os.path.abspath(audio_path)
        except: sys.stderr.write("Lưu ý: Không thể trích xuất audio.\\n")
        print(json.dumps({"frame_folder": os.path.abspath(frame_dir), "audio_path": audio_out_path}))
    except Exception as e: print(json.dumps({"error": "Lỗi FFmpeg", "details": str(e)}))
if __name__ == "__main__": main()"""),
    ("tool/frames_to_video.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os; from ffmpy import FFmpeg
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--frame_folder', required=True); parser.add_argument('--audio_path', default=None); parser.add_argument('--metadata_json', required=True); parser.add_argument('--output', required=True); args = parser.parse_args()
    try: fps = json.loads(args.metadata_json).get('fps')
    except: sys.exit(json.dumps({"error": "JSON metadata không hợp lệ"}))
    if not fps: sys.exit(json.dumps({"error": "Không tìm thấy 'fps'"}))
    os.makedirs(os.path.dirname(args.output), exist_ok=True); pattern = os.path.join(args.frame_folder, '%08d.png'); inputs = {pattern: ['-framerate', str(fps)]}
    if args.audio_path and os.path.exists(args.audio_path): inputs[args.audio_path] = None
    opts = ['-c:v','libx264','-pix_fmt','yuv420p']
    if args.audio_path and os.path.exists(args.audio_path): opts.extend(['-c:a','aac','-shortest'])
    ff = FFmpeg(global_options=['-y'], inputs=inputs, outputs={args.output: opts})
    try: ff.run(stderr=sys.stderr); print(json.dumps({"output_path": os.path.abspath(args.output)}))
    except Exception as e: print(json.dumps({"error": "Lỗi FFmpeg", "details": str(e)}))
if __name__ == "__main__": main()"""),
    ("tool/merge_video.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os; from ffmpy import FFmpeg
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--input_list_file', required=True); parser.add_argument('--output', required=True); args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    ff = FFmpeg(global_options=['-y'], inputs={args.input_list_file: ['-f', 'concat', '-safe', '0']}, outputs={args.output: ['-c', 'copy']})
    try: ff.run(stderr=sys.stderr); print(json.dumps({"final_video_path": os.path.abspath(args.output)}))
    except Exception as e: print(json.dumps({"error": "Lỗi FFmpeg", "details": str(e)}))
if __name__ == "__main__": main()"""),
    ("script_AI/images_processing.py", """# -*- coding: utf-8 -*-
import torch, cv2, numpy as np; from torchvision import transforms
def load_image_to_tensor(p, size=(512,512), norm=True):
    img=cv2.imread(p); orig_s=img.shape[:2]; img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tl=[transforms.ToTensor(),transforms.Resize(size,antialias=True)]
    if norm: tl.append(transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]))
    return transforms.Compose(tl)(img).unsqueeze(0),orig_s
def save_tensor_as_image(t,op,orig_s,is_inpaint=False):
    t=t.detach().cpu().squeeze(0); t=transforms.Resize(orig_s,antialias=True)(t)
    if is_inpaint: t=t*0.5+0.5
    img=np.clip(t.permute(1,2,0).numpy()*255,0,255).astype(np.uint8)
    cv2.imwrite(op,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
def load_mask_to_tensor(p,size=(512,512)):
    m=cv2.imread(p,cv2.IMREAD_GRAYSCALE); _,m=cv2.threshold(m,127,255,cv2.THRESH_BINARY)
    return transforms.Compose([transforms.ToTensor(),transforms.Resize(size,antialias=True)])(m).unsqueeze(0)"""),
    ("script_AI/model_loader.py", """# -*- coding: utf-8 -*-
import torch, os, sys; SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.append(os.path.dirname(SCRIPT_DIR))
from script_AI.models import AddYouknowModel, CleanYouknowModel
def load_model(name,pth,dev='cpu'):
    if name in ['add_youknow','mosaic_position']: model=AddYouknowModel()
    elif name == 'clean_youknow': model=CleanYouknowModel()
    else: raise ValueError(f"Model không hỗ trợ: {name}")
    sd=torch.load(pth, map_location=torch.device(dev));
    if 'state_dict' in sd: sd=sd['state_dict']
    model.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}); return model.to(dev).eval()"""),
    ("script_AI/models.py", """# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F
class ResNet18(nn.Module):
    def __init__(self): super().__init__(); resnet=torch.hub.load('pytorch/vision:v0.10.0','resnet18',pretrained=True); self.f=nn.Sequential(*list(resnet.children())[:-2])
    def forward(self,x): return self.f(x)
class AddYouknowModel(nn.Module):
    def __init__(self, nc=1):
        super().__init__(); self.backbone=ResNet18()
        self.decoder=nn.Sequential(nn.Conv2d(512,256,3,padding=1),nn.ReLU(True),nn.Upsample(scale_factor=2), nn.Conv2d(256,128,3,padding=1),nn.ReLU(True),nn.Upsample(scale_factor=2), nn.Conv2d(128,64,3,padding=1),nn.ReLU(True),nn.Upsample(scale_factor=2), nn.Conv2d(64,32,3,padding=1),nn.ReLU(True),nn.Upsample(scale_factor=2), nn.Conv2d(32,nc,3,padding=1))
    def forward(self,x): return F.interpolate(self.decoder(self.backbone(x)), size=x.shape[2:], mode='bilinear', align_corners=True)
class PConv(nn.Module):
    def __init__(self,i,o,bn=True,sample='none',activ='relu',bias=False):
        super().__init__(); self.s=None;
        if sample=='down': self.s=nn.MaxPool2d(2,2)
        elif sample=='up': self.s=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.c=nn.Conv2d(i,o,3,1,1,bias=bias); self.mc=nn.Conv2d(i,o,3,1,1,bias=False); torch.nn.init.constant_(self.mc.weight,1.0)
        for p in self.mc.parameters(): p.requires_grad=False
        self.bn=nn.BatchNorm2d(o) if bn else None; self.a=nn.ReLU(True) if activ=='relu' else nn.LeakyReLU(0.2,True)
    def forward(self,x,m):
        if self.s: x,m=self.s(x),self.s(m)
        with torch.no_grad(): mr=1/(self.mc(m)+1e-8)
        o=self.c(x*m)*mr;
        if self.bn:o=self.bn(o)
        return self.a(o),m
class CleanYouknowModel(nn.Module):
    def __init__(self):
        super().__init__(); self.e1=PConv(4,64,bn=False,sample='down'); self.e2=PConv(64,128,sample='down'); self.e3=PConv(128,256,sample='down'); self.e4=PConv(256,512,sample='down'); self.d1=PConv(512+256,256,activ='leaky',sample='up'); self.d2=PConv(256+128,128,activ='leaky',sample='up'); self.d3=PConv(128+64,64,activ='leaky',sample='up'); self.d4=PConv(64+4,3,bn=False,activ='leaky'); self.f=nn.Conv2d(3,3,1)
    def forward(self,x,m):
        e1,m1=self.e1(x,m);e2,m2=self.e2(e1,m1);e3,m3=self.e3(e2,m2);e4,m4=self.e4(e3,m3); d1,_=self.d1(torch.cat([e4,e3],1),torch.cat([m4,m3],1));d2,_=self.d2(torch.cat([d1,e2],1),torch.cat([m2,m2],1)); d3,_=self.d3(torch.cat([d2,e1],1),torch.cat([m2,m1],1));d4,_=self.d4(torch.cat([d3,x],1),torch.cat([m1,m],1)); return torch.tanh(self.f(d4))"""),
    ("script_AI/run/run_add_youknow.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os,torch; from tqdm import tqdm; SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__)); sys.path.append(os.path.dirname(SCRIPT_DIR)); from script_AI.model_loader import load_model; from script_AI.images_processing import load_image_to_tensor,save_tensor_as_image
def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--output_dir',required=True);a=p.parse_args(); dev='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_model('add_youknow',os.path.join(os.path.dirname(SCRIPT_DIR), 'pre_trained_models/add_youknow.pth'),dev)
    os.makedirs(a.output_dir,exist_ok=True)
    for fn in tqdm(os.listdir(a.input_dir),file=sys.stderr):
        try: t,s=load_image_to_tensor(os.path.join(a.input_dir,fn)); mt=torch.sigmoid(model(t.to(dev))); save_tensor_as_image(mt,os.path.join(a.output_dir,fn),s)
        except Exception as e:sys.stderr.write(f"Lỗi file {fn}: {e}\\n")
    print(json.dumps({"processed_frame_folder":os.path.abspath(a.output_dir)}))
if __name__=="__main__":main()"""),
    ("script_AI/run/run_mosaic_position.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os,torch; from tqdm import tqdm; SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__)); sys.path.append(os.path.dirname(SCRIPT_DIR)); from script_AI.model_loader import load_model; from script_AI.images_processing import load_image_to_tensor,save_tensor_as_image
def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--output_dir',required=True);a=p.parse_args(); dev='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_model('mosaic_position',os.path.join(os.path.dirname(SCRIPT_DIR),'pre_trained_models/mosaic_position.pth'),dev)
    os.makedirs(a.output_dir,exist_ok=True)
    for fn in tqdm(os.listdir(a.input_dir),file=sys.stderr):
        try: t,s=load_image_to_tensor(os.path.join(a.input_dir,fn)); mt=torch.sigmoid(model(t.to(dev))); save_tensor_as_image(mt,os.path.join(a.output_dir,fn),s)
        except Exception as e:sys.stderr.write(f"Lỗi file {fn}: {e}\\n")
    print(json.dumps({"processed_frame_folder":os.path.abspath(a.output_dir)}))
if __name__=="__main__":main()"""),
    ("script_AI/run/run_clean_youknow.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os,torch; from tqdm import tqdm; SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__)); sys.path.append(os.path.dirname(SCRIPT_DIR)); from script_AI.model_loader import load_model; from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor,save_tensor_as_image
def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--mask_dir',required=True);p.add_argument('--output_dir',required=True);a=p.parse_args(); dev='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_model('clean_youknow',os.path.join(os.path.dirname(SCRIPT_DIR),'pre_trained_models/clean_youknow.pth'),dev)
    os.makedirs(a.output_dir,exist_ok=True)
    for fn in tqdm(os.listdir(a.input_dir),file=sys.stderr):
        mp=os.path.join(a.mask_dir,fn);
        if not os.path.exists(mp):continue
        try:
            it,s=load_image_to_tensor(os.path.join(a.input_dir,fn),norm=False);it_norm=it*2-1
            mt=load_mask_to_tensor(mp);it_norm,mt=it_norm.to(dev),mt.to(dev)
            with torch.no_grad():out=model(torch.cat((it_norm,mt),dim=1),mt)
            final=(out*mt)+(it_norm*(1-mt))
            save_tensor_as_image(final,os.path.join(a.output_dir,fn),s,is_inpaint=True)
        except Exception as e:sys.stderr.write(f"Lỗi file {fn}: {e}\\n")
    print(json.dumps({"processed_frame_folder":os.path.abspath(a.output_dir)}))
if __name__=="__main__":main()"""),
    ("script_AI/train/train_segmentation.py", """# -*- coding: utf-8 -*-
import torch,os,sys;import torch.nn as nn;import torch.optim as optim;from torch.utils.data import Dataset,DataLoader;from tqdm import tqdm;sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor
class SegDS(Dataset):
    def __init__(self,id,md):self.id,self.md,self.f=id,md,sorted(os.listdir(id))
    def __len__(self):return len(self.f)
    def __getitem__(self,i):img,_=load_image_to_tensor(os.path.join(self.id,self.f[i]));msk=load_mask_to_tensor(os.path.join(self.md,self.f[i]));return img.squeeze(0),msk.squeeze(0)
def train(m,d,l,o,c):
    m.train()
    for i,k in tqdm(l,file=sys.stderr):i,k=i.to(d),k.to(d);o.zero_grad();loss=c(m(i),k);loss.backward();o.step()
def run_training_session(m,dp,e,lr,sp):
    dev='cuda' if torch.cuda.is_available() else 'cpu';m.to(dev);ds=SegDS(os.path.join(dp,'images'),os.path.join(dp,'masks'));l=DataLoader(ds,batch_size=4,shuffle=True,num_workers=2);opt=optim.Adam(m.parameters(),lr=lr);crit=nn.BCEWithLogitsLoss()
    for ep in range(e):sys.stderr.write(f"Epoch {ep+1}/{e}\\n");train(m,dev,l,opt,crit);torch.save(m.state_dict(),sp)"""),
    ("script_AI/train/train_clean_youknow.py", """# -*- coding: utf-8 -*-
import torch,os,sys;import torch.nn as nn;import torch.optim as optim;from torch.utils.data import Dataset,DataLoader;from tqdm import tqdm;sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor
class InpaintDS(Dataset):
    def __init__(self,dp):self.gtd,self.md,self.mskd=os.path.join(dp,'original_images'),os.path.join(dp,'mosaiced_images'),os.path.join(dp,'mosaic_masks');self.f=sorted(os.listdir(self.gtd))
    def __len__(self):return len(self.f)
    def __getitem__(self,i):gt,_=load_image_to_tensor(os.path.join(self.gtd,self.f[i]),norm=False);mo,_=load_image_to_tensor(os.path.join(self.md,self.f[i]),norm=False);msk=load_mask_to_tensor(os.path.join(self.mskd,self.f[i]));return gt.squeeze(0),mo.squeeze(0),msk.squeeze(0)
def train(m,d,l,o,c):
    m.train()
    for gt,mo,msk in tqdm(l,file=sys.stderr):gt,mo,msk=gt.to(d),mo.to(d),msk.to(d);mi=torch.cat((mo*2-1,msk),1);o.zero_grad();out=m(mi,msk);loss=c(out*msk,gt*msk);loss.backward();o.step()
def run_training_session_inpaint(m,dp,e,lr,sp):
    dev='cuda' if torch.cuda.is_available() else 'cpu';m.to(dev);ds=InpaintDS(dp);l=DataLoader(ds,batch_size=2,shuffle=True,num_workers=2);opt=optim.Adam(m.parameters(),lr=lr);crit=nn.L1Loss()
    for ep in range(e):sys.stderr.write(f"Epoch {ep+1}/{e}\\n");train(m,dev,l,opt,crit);torch.save(m.state_dict(),sp)"""),
    # --- READMEs ---
    ("README.md", """# Dự án UnOrCensored
Bộ công cụ xử lý video và hình ảnh bằng AI.
## Cài đặt: `python setup.py`
## Sử dụng:
- Xử lý: `python run.py --file_path <path> --task_name <task>`
- Huấn luyện: `python train.py --pre_trained_models_name <model_name>`"""),
    ("data/README.md", "# Thư mục Dữ liệu (data)\\nChứa dữ liệu để huấn luyện model."),
    ("data/add_youknow/README.md", "# Dữ liệu `add_youknow`\\n- `images/`: Ảnh gốc.\\n- `masks/`: Mask tương ứng."),
    ("data/add_youknow/images/README.md", "# Thư mục `images`"),
    ("data/add_youknow/masks/README.md", "# Thư mục `masks`"),
    ("data/clean_youknow/README.md", "# Dữ liệu `clean_youknow`\\n- `original_images/`: Ảnh gốc.\\n- `mosaiced_images/`: Ảnh đã che.\\n- `mosaic_masks/`: Mask vị trí."),
    ("data/clean_youknow/original_images/README.md", "# Thư mục `original_images`"),
    ("data/clean_youknow/mosaiced_images/README.md", "# Thư mục `mosaiced_images`"),
    ("data/clean_youknow/mosaic_masks/README.md", "# Thư mục `mosaic_masks`"),
    ("data/mosaic_position/README.md", "# Dữ liệu `mosaic_position`\\n- `mosaiced_images/`: Ảnh đã che.\\n- `mosaic_masks/`: Mask thực tế."),
    ("data/mosaic_position/mosaiced_images/README.md", "# Thư mục `mosaiced_images`"),
    ("data/mosaic_position/mosaic_masks/README.md", "# Thư mục `mosaic_masks`"),
    ("output/README.md", "# Thư mục `output`\\nLưu kết quả từ `run.py`."),
    ("pre_trained_models/README.md", "# Thư mục `pre_trained_models`\\nChứa các file trọng số `.pth`."),
    ("script_AI/README.md", "# Thư mục `script_AI`\\nChứa mã nguồn AI."),
    ("script_AI/run/README.md", "# Thư mục `run`\\nChứa các script thực thi inference."),
    ("script_AI/train/README.md", "# Thư mục `train`\\nChứa các module logic huấn luyện."),
    ("tool/README.md", "# Thư mục `tool`\\nChứa các script công cụ xử lý media.")
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



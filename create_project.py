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
torchaudio==2.6.0+cu124"""
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
    print(">>> Bước 1: Bắt đầu cài đặt các thư viện...")
    if not os.path.isfile(REQUIREMENTS_FILE):
        print(f">>> [Lỗi] Không tìm thấy file requirements.txt tại: {REQUIREMENTS_FILE}", file=sys.stderr)
        sys.exit(1)
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE], check=True, capture_output=True, text=True, encoding='utf-8')
        print(">>> [Thành công] Đã cài đặt xong tất cả các thư viện.")
    except subprocess.CalledProcessError as e:
        print(f">>> [Lỗi] Cài đặt thất bại:\\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def download_models():
    print(f"\\n>>> Bước 2: Bắt đầu tải models vào '{PRE_TRAINED_MODELS_DIR}'...")
    try:
        gdown.download_folder(id=MODEL_FOLDER_ID, output=PRE_TRAINED_MODELS_DIR, quiet=False, use_cookies=False)
        print(">>> [Thành công] Đã tải xong các model.")
    except Exception as e:
        print(f">>> [Lỗi] Tải model thất bại: {e}", file=sys.stderr)
        sys.exit(1)

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

# --- Lấy đường dẫn tuyệt đối của thư mục chứa script này ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Định nghĩa các đường dẫn tuyệt đối ---
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
    
    ai_script = f"run_{args.task_name}.py"
    cmd = ["python", os.path.join(SCRIPT_AI_RUN_DIR, ai_script), "--input_dir", input_dir, "--output_dir", output_dir]
    
    if args.task_name == 'clean_youknow':
        sys.stderr.write("Lưu ý: Luồng clean_youknow cho ảnh yêu cầu một thư mục mask tương ứng. Hãy đảm bảo cung cấp nó nếu cần.\\n")

    run_command(cmd)
    
    final_output_dir = args.folder_path or OUTPUT_DIR
    os.makedirs(final_output_dir, exist_ok=True)
    shutil.move(os.path.join(output_dir, os.path.basename(args.file_path)), final_output_dir)
    sys.stderr.write(f"Đã lưu kết quả vào: {final_output_dir}\\n")

def handle_video(args, temp_dir):
    current_video_path = args.file_path
    
    info = run_command(["python", os.path.join(TOOL_DIR, "get_file_info.py"), "--input", current_video_path])
    metadata = info.get("metadata", {})
    metadata_json = json.dumps(metadata)
    
    duration = run_command(["python", os.path.join(TOOL_DIR, "duration_split.py"), "--input", current_video_path, "--fps", str(metadata.get('fps', 30))]).get("optimal_chunk_duration", 300)
    
    chunks_dir = os.path.join(temp_dir, "chunks")
    chunk_paths = run_command(["python", os.path.join(TOOL_DIR, "split_video.py"), "--input", current_video_path, "--duration", str(duration), "--output_dir", chunks_dir]).get("chunk_paths", [])
    
    processed_chunks = []
    for i, chunk_path in enumerate(chunk_paths):
        sys.stderr.write(f"\\n>>> Đang xử lý chunk {i+1}/{len(chunk_paths)}...\\n")
        chunk_proc_dir = os.path.join(temp_dir, f"chunk_{i}")
        
        frames_info = run_command(["python", os.path.join(TOOL_DIR, "video_to_frames.py"), "--input", chunk_path, "--output_dir", chunk_proc_dir])
        
        processed_frames_dir = os.path.join(chunk_proc_dir, "processed")
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, f"run_{args.task_name}.py"), "--input_dir", frames_info['frame_folder'], "--output_dir", processed_frames_dir])
        
        proc_chunk_path_cmd = [
            "python", os.path.join(TOOL_DIR, "frames_to_video.py"),
            "--frame_folder", processed_frames_dir,
            "--metadata_json", metadata_json,
            "--output", os.path.join(chunk_proc_dir, "out.mp4")
        ]
        if frames_info.get('audio_path'):
            proc_chunk_path_cmd.extend(["--audio_path", frames_info.get('audio_path')])
        
        proc_chunk_path = run_command(proc_chunk_path_cmd).get("output_path")
        processed_chunks.append(proc_chunk_path)

    list_file = os.path.join(temp_dir, "mergelist.txt")
    with open(list_file, "w", encoding='utf-8') as f:
        for p in processed_chunks:
            # SỬA LỖI: Gán kết quả của replace cho biến mới trước khi dùng trong f-string
            clean_path = p.replace('\\\\', '/')
            f.write(f"file '{clean_path}'\\n")
        
    final_path = os.path.join(args.folder_path or OUTPUT_DIR, f"final_{os.path.basename(args.file_path)}")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    run_command(["python", os.path.join(TOOL_DIR, "merge_video.py"), "--input_list_file", list_file, "--output", final_path])

def main():
    parser = argparse.ArgumentParser(description="Script điều phối chính cho dự án UnOrCensored.")
    parser.add_argument('--file_path', required=True, help="Đường dẫn file đầu vào.")
    parser.add_argument('--task_name', required=True, choices=['add_youknow', 'clean_youknow', 'mosaic_position'], help="Tên tác vụ.")
    parser.add_argument('--folder_path', help="Thư mục đầu ra.")
    args = parser.parse_args()

    temp_dir = os.path.join(TMP_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(temp_dir, exist_ok=True)
    try:
        file_type = run_command(["python", os.path.join(TOOL_DIR, "get_file_type.py"), "--input", args.file_path]).get("file_type")
        if file_type == 'image': handle_image(args, temp_dir)
        elif file_type == 'video': handle_video(args, temp_dir)
        else: sys.stderr.write("Loại file không được hỗ trợ.\\n")
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

# Thêm thư mục gốc vào path để có thể import các module khác
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from script_AI.model_loader import load_model
from script_AI.train import train_segmentation, train_clean_youknow

def main():
    parser = argparse.ArgumentParser(description="Script điều phối chính cho việc tinh chỉnh model.")
    parser.add_argument('--pre_trained_models_name', required=True, choices=['add_youknow', 'mosaic_position', 'clean_youknow'], help="Tên model cần tinh chỉnh.")
    parser.add_argument('--folder_path', help="Đường dẫn thư mục dữ liệu. Mặc định: 'data/<model_name>'.")
    parser.add_argument('--epochs', type=int, default=10, help="Số epochs.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    args = parser.parse_args()

    data_path = args.folder_path or os.path.join(SCRIPT_DIR, 'data', args.pre_trained_models_name)
    if not os.path.isdir(data_path):
        sys.stderr.write(f"[Lỗi] Thư mục dữ liệu không tồn tại: {data_path}\\n")
        sys.exit(1)

    base_model_path = os.path.join(SCRIPT_DIR, 'pre_trained_models', f"{args.pre_trained_models_name}.pth")
    model = load_model(args.pre_trained_models_name, base_model_path)
    
    save_path = os.path.join(SCRIPT_DIR, 'pre_trained_models', f"{args.pre_trained_models_name}_finetuned_{datetime.datetime.now().strftime('%Y%m%d')}.pth")

    try:
        if args.pre_trained_models_name in ['add_youknow', 'mosaic_position']:
            train_segmentation.run_training_session(model, data_path, args.epochs, args.lr, save_path)
        elif args.pre_trained_models_name == 'clean_youknow':
            train_clean_youknow.run_training_session_inpaint(model, data_path, args.epochs, args.lr, save_path)
    except Exception as e:
        sys.stderr.write(f"\\n[Lỗi] Huấn luyện thất bại: {e}\\n")
        sys.exit(1)
        
    sys.stderr.write(f"\\nHoàn tất! Model đã tinh chỉnh được lưu tại: {save_path}\\n")

if __name__ == '__main__':
    main()"""
    ),
    # --- tool/ ---
    ("tool/get_file_type.py", """# -*- coding: utf-8 -*-
import os, argparse, json, sys

def get_file_type(file_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    ext = os.path.splitext(file_path)[1].lower()
    if ext in video_extensions: return {"file_type": "video"}
    if ext in image_extensions: return {"file_type": "image"}
    return {"file_type": "unknown"}

def main():
    parser = argparse.ArgumentParser(description="Xác định loại file.")
    parser.add_argument('--input', required=True, help="Đường dẫn file.")
    args = parser.parse_args()
    print(json.dumps(get_file_type(args.input)))

if __name__ == "__main__":
    main()"""),
    ("tool/get_file_info.py", """# -*- coding: utf-8 -*-
import cv2, argparse, json, sys

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Không thể mở video: {video_path}"}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        metadata = {
            "fps": fps,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0,
            "codec_video": "".join([chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> 8 * i) & 0xFF) for i in range(4)]).strip()
        }
        return {"metadata": metadata}
    finally:
        cap.release()

def main():
    parser = argparse.ArgumentParser(description="Trích xuất metadata video.")
    parser.add_argument('--input', required=True, help="Đường dẫn video.")
    args = parser.parse_args()
    print(json.dumps(get_video_info(args.input)))

if __name__ == "__main__":
    main()"""),
    ("tool/cut_video.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os
from ffmpy import FFmpeg, FFRuntimeError

def cut_video(input_path, output_path, start_time, end_time):
    output_options = ['-c', 'copy']
    if end_time and end_time != "00:00:00":
        output_options.extend(['-to', end_time])
    ff = FFmpeg(global_options=['-y'], inputs={input_path: ['-ss', start_time]}, outputs={output_path: output_options})
    try:
        ff.run(stdout=sys.stderr, stderr=sys.stderr)
        return {"output_path": os.path.abspath(output_path)}
    except FFRuntimeError as e:
        return {"error": "Lỗi FFmpeg khi cắt video", "details": e.stderr}

def main():
    parser = argparse.ArgumentParser(description="Cắt video theo thời gian.")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', default=None)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(json.dumps(cut_video(args.input, args.output, args.start, args.end)))

if __name__ == "__main__":
    main()"""),
    ("tool/duration_split.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os,psutil
from ffmpy import FFmpeg

def get_optimal_chunk_duration(video_path, fps):
    temp_dir = "tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_frame_path = os.path.join(temp_dir, f"temp_frame_{os.path.basename(video_path)}.png")
    ff = FFmpeg(global_options=['-y'], inputs={video_path:['-ss','00:00:01']}, outputs={temp_frame_path:['-vframes','1']})
    try:
        ff.run(stderr=sys.stderr)
        exact_frame_size = os.path.getsize(temp_frame_path)
        data_rate = exact_frame_size * 3 * fps
        usable_space = psutil.disk_usage(os.path.abspath('.')).free * 0.8
        if data_rate == 0: return {"optimal_chunk_duration": 300}
        return {"optimal_chunk_duration": int(usable_space / data_rate)}
    except Exception:
        return {"optimal_chunk_duration": 300} # Fallback an toàn
    finally:
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)

def main():
    parser = argparse.ArgumentParser(description="Tính thời lượng chunk tối ưu.")
    parser.add_argument('--input', required=True)
    parser.add_argument('--fps', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(get_optimal_chunk_duration(args.input, args.fps)))

if __name__ == "__main__":
    main()"""),
    ("tool/split_video.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os
from ffmpy import FFmpeg, FFRuntimeError

def split_video(input_path, output_dir, duration):
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, f"chunk_%05d{os.path.splitext(input_path)[1]}")
    ff = FFmpeg(global_options=['-y'], inputs={input_path: None}, outputs={output_pattern: ['-c', 'copy', '-map', '0', '-segment_time', str(duration), '-f', 'segment', '-reset_timestamps', '1']})
    try:
        ff.run(stdout=sys.stderr, stderr=sys.stderr)
        chunk_files = sorted([os.path.abspath(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.startswith('chunk_')])
        return {"chunk_paths": chunk_files}
    except FFRuntimeError as e:
        return {"error": "Lỗi FFmpeg khi chia video", "details": e.stderr}

def main():
    parser = argparse.ArgumentParser(description="Chia video thành chunks.")
    parser.add_argument('--input', required=True)
    parser.add_argument('--duration', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    print(json.dumps(split_video(args.input, args.output_dir, args.duration)))

if __name__ == "__main__":
    main()"""),
    ("tool/video_to_frames.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os
from ffmpy import FFmpeg, FFRuntimeError

def video_to_frames(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_folder_path = os.path.join(output_dir, 'frames')
    os.makedirs(frame_folder_path, exist_ok=True)
    audio_path = os.path.join(output_dir, "audio.aac")
    frame_pattern = os.path.join(frame_folder_path, '%08d.png')
    
    ff_frames = FFmpeg(global_options=['-y'], inputs={input_path: None}, outputs={frame_pattern: ['-qscale:v', '2']})
    ff_audio = FFmpeg(global_options=['-y'], inputs={input_path: None}, outputs={audio_path: ['-vn', '-acodec', 'copy']})
    
    try:
        ff_frames.run(stdout=sys.stderr, stderr=sys.stderr)
        audio_result_path = None
        try:
            ff_audio.run(stdout=sys.stderr, stderr=sys.stderr)
            audio_result_path = os.path.abspath(audio_path)
        except FFRuntimeError:
            sys.stderr.write("Lưu ý: Không thể trích xuất audio.\\n")
        return {"frame_folder": os.path.abspath(frame_folder_path), "audio_path": audio_result_path}
    except FFRuntimeError as e:
        return {"error": "Lỗi FFmpeg khi trích xuất frames", "details": e.stderr}

def main():
    parser = argparse.ArgumentParser(description="Trích xuất frames và audio.")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    print(json.dumps(video_to_frames(args.input, args.output_dir)))

if __name__ == "__main__":
    main()"""),
    ("tool/frames_to_video.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os
from ffmpy import FFmpeg, FFRuntimeError

def frames_to_video(frame_folder, audio_path, metadata_json, output_path):
    try:
        fps = json.loads(metadata_json).get('fps')
    except json.JSONDecodeError:
        return {"error": "JSON metadata không hợp lệ"}
    if not fps:
        return {"error": "Không tìm thấy 'fps' trong metadata JSON."}

    frame_pattern = os.path.join(frame_folder, '%08d.png')
    input_files = {frame_pattern: ['-framerate', str(fps)]}
    if audio_path and os.path.exists(audio_path):
        input_files[audio_path] = None
    
    output_options = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p']
    if audio_path and os.path.exists(audio_path):
        output_options.extend(['-c:a', 'aac', '-shortest'])
    
    ff = FFmpeg(global_options=['-y'], inputs=input_files, outputs={output_path: output_options})
    try:
        ff.run(stdout=sys.stderr, stderr=sys.stderr)
        return {"output_path": os.path.abspath(output_path)}
    except FFRuntimeError as e:
        return {"error": "Lỗi FFmpeg khi ghép frames", "details": e.stderr}

def main():
    parser = argparse.ArgumentParser(description="Ghép frames thành video.")
    parser.add_argument('--frame_folder', required=True)
    parser.add_argument('--audio_path', default=None)
    parser.add_argument('--metadata_json', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(json.dumps(frames_to_video(args.frame_folder, args.audio_path, args.metadata_json, args.output)))

if __name__ == "__main__":
    main()"""),
    ("tool/merge_video.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os
from ffmpy import FFmpeg, FFRuntimeError

def merge_videos(input_list_file, output_path):
    ff = FFmpeg(global_options=['-y'], inputs={input_list_file: ['-f', 'concat', '-safe', '0']}, outputs={output_path: ['-c', 'copy']})
    try:
        ff.run(stdout=sys.stderr, stderr=sys.stderr)
        return {"final_video_path": os.path.abspath(output_path)}
    except FFRuntimeError as e:
        return {"error": "Lỗi FFmpeg khi nối video", "details": e.stderr}

def main():
    parser = argparse.ArgumentParser(description="Nối video từ file danh sách.")
    parser.add_argument('--input_list_file', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(json.dumps(merge_videos(args.input_list_file, args.output)))

if __name__ == "__main__":
    main()"""),
    ("script_AI/images_processing.py", """# -*- coding: utf-8 -*-
import torch, cv2, numpy as np
from torchvision import transforms
def load_image_to_tensor(p, target_size=(512,512), normalize=True):
    img=cv2.imread(p)
    if img is None: raise FileNotFoundError(f"Không thể mở: {p}")
    orig_s=img.shape[:2]
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    tl=[transforms.ToTensor(),transforms.Resize(target_size,antialias=True)]
    if normalize: tl.append(transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]))
    return transforms.Compose(tl)(img).unsqueeze(0),orig_s
def save_tensor_as_image(t,op,orig_s,is_inpainting=False):
    t=t.detach().cpu().squeeze(0)
    t=transforms.Resize(orig_s,antialias=True)(t)
    if is_inpainting: t=t*0.5+0.5
    img_np=t.permute(1,2,0).numpy()
    img_np=np.clip(img_np*255,0,255).astype(np.uint8)
    cv2.imwrite(op,cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR))
def load_mask_to_tensor(p,target_size=(512,512)):
    m=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(f"Không thể mở: {p}")
    _,m=cv2.threshold(m,127,255,cv2.THRESH_BINARY)
    return transforms.Compose([transforms.ToTensor(),transforms.Resize(target_size,antialias=True)])(m).unsqueeze(0)"""),
    ("script_AI/model_loader.py", """# -*- coding: utf-8 -*-
import torch, os, sys
# Đảm bảo có thể import từ các thư mục khác
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

try: from script_AI.models import AddYouknowModel, CleanYouknowModel
except ImportError: from models import AddYouknowModel, CleanYouknowModel
def load_model(name, pth, dev='cpu'):
    sys.stderr.write(f"Đang tải model '{name}'...\\n")
    if name in ['add_youknow','mosaic_position']: model=AddYouknowModel(num_classes=1)
    elif name == 'clean_youknow': model=CleanYouknowModel()
    else: raise ValueError(f"Model không hỗ trợ: {name}")
    if not os.path.exists(pth): raise FileNotFoundError(f"Không tìm thấy: {pth}")
    try:
        sd=torch.load(pth, map_location=torch.device(dev))
        if 'state_dict' in sd: sd=sd['state_dict']
        if 'model' in sd: sd=sd['model']
        new_sd={k.replace('module.',''):v for k,v in sd.items()}
        model.load_state_dict(new_sd)
    except Exception as e: raise RuntimeError(f"Lỗi nạp trọng số: {e}")
    model.to(dev).eval()
    sys.stderr.write(f"Tải model '{name}' thành công.\\n")
    return model"""),
    ("script_AI/models.py", """# -*- coding: utf-8 -*-
import torch, torch.nn as nn, torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self,i,o,**k): super().__init__(); self.r=nn.ReLU(True); self.c=nn.Conv2d(i,o,bias=False,**k); self.b=nn.BatchNorm2d(o)
    def forward(self,x): return self.r(self.b(self.c(x)))
class ARM(nn.Module):
    def __init__(self,i,o): super().__init__(); self.c=ConvBlock(i,o,kernel_size=3,padding=1); self.a=nn.Sequential(nn.AdaptiveAvgPool2d(1),ConvBlock(o,o,kernel_size=1),nn.Sigmoid())
    def forward(self,x): f=self.c(x); return f*self.a(f)
class FFM(nn.Module):
    def __init__(self,i,o): super().__init__(); self.c=ConvBlock(i,o,kernel_size=1); self.a=nn.Sequential(nn.AdaptiveAvgPool2d(1),ConvBlock(o,o//4,kernel_size=1),ConvBlock(o//4,o,kernel_size=1),nn.Sigmoid())
    def forward(self,xs,xc): x=torch.cat([xs,xc],1); f=self.c(x); return f+(f*self.a(f))
class ResNet18(nn.Module):
    def __init__(self): super().__init__(); resnet=torch.hub.load('pytorch/vision:v0.10.0','resnet18',pretrained=True); self.conv1=resnet.conv1; self.bn1=resnet.bn1; self.relu=resnet.relu; self.maxpool=resnet.maxpool; self.layer1=resnet.layer1; self.layer2=resnet.layer2; self.layer3=resnet.layer3; self.layer4=resnet.layer4
    def forward(self,x): x=self.relu(self.bn1(self.conv1(x))); x=self.maxpool(x); c3=self.layer3(self.layer2(self.layer1(x))); c4=self.layer4(c3); return c3,c4
class AddYouknowModel(nn.Module):
    def __init__(self,nc=1): super().__init__(); self.cp=ResNet18(); self.sp=nn.Sequential(ConvBlock(3,64,kernel_size=7,stride=2,padding=3),ConvBlock(64,64,kernel_size=3,stride=2,padding=1),ConvBlock(64,128,kernel_size=1)); self.arm32=ARM(256,128); self.arm16=ARM(512,128); self.ffm=FFM(256,256); self.cout=nn.Conv2d(256,nc,1)
    def forward(self,x): H,W=x.size()[2:]; xs=self.sp(x); c3,c4=self.cp(x); c3a=self.arm32(c3); c4a=self.arm16(c4); c4a_up=F.interpolate(c4a,size=c3a.size()[2:],mode='bilinear',align_corners=True); cf=c3a+c4a_up; fused=self.ffm(xs,F.interpolate(cf,size=xs.size()[2:],mode='bilinear',align_corners=True)); return F.interpolate(self.cout(fused),size=(H,W),mode='bilinear',align_corners=True)
class PConv(nn.Module):
    def __init__(self,i,o,bn=True,sample='none',activ='relu',bias=False):
        super().__init__(); self.sample=sample
        if sample=='down': self.s=nn.MaxPool2d(2,2)
        elif sample=='up': self.s=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.c=nn.Conv2d(i,o,3,1,1,bias=bias); self.mc=nn.Conv2d(i,o,3,1,1,bias=False); torch.nn.init.constant_(self.mc.weight,1.0);
        for p in self.mc.parameters(): p.requires_grad=False
        self.bn=nn.BatchNorm2d(o) if bn else None; self.a=nn.ReLU(True) if activ=='relu' else nn.LeakyReLU(0.2,True)
    def forward(self,x,m):
        if hasattr(self,'s'): x,m=self.s(x),self.s(m)
        with torch.no_grad(): mr=512*512/(self.mc(m)+1e-8)
        o=self.c(x*m)*mr
        if self.bn: o=self.bn(o)
        return self.a(o),m
class CleanYouknowModel(nn.Module):
    def __init__(self): super().__init__(); self.e1=PConv(4,64,bn=False,sample='down'); self.e2=PConv(64,128,sample='down'); self.e3=PConv(128,256,sample='down'); self.e4=PConv(256,512,sample='down'); self.d1=PConv(512+256,256,activ='leaky',sample='up'); self.d2=PConv(256+128,128,activ='leaky',sample='up'); self.d3=PConv(128+64,64,activ='leaky',sample='up'); self.d4=PConv(64+4,3,bn=False,activ='leaky'); self.f=nn.Conv2d(3,3,1)
    def forward(self,x,m):
        e1,m1=self.e1(x,m); e2,m2=self.e2(e1,m1); e3,m3=self.e3(e2,m2); e4,m4=self.e4(e3,m3);
        d1,_=self.d1(torch.cat([e4,e3],1),torch.cat([m4,m3],1)); d2,_=self.d2(torch.cat([d1,e2],1),torch.cat([m2,m2],1)); d3,_=self.d3(torch.cat([d2,e1],1),torch.cat([m2,m1],1)); d4,_=self.d4(torch.cat([d3,x],1),torch.cat([m1,m],1)); return self.f(d4)"""),
    ("script_AI/run/run_add_youknow.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os,torch; from tqdm import tqdm; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from script_AI.model_loader import load_model; from script_AI.images_processing import load_image_to_tensor,save_tensor_as_image
def process(indir,outdir,dev):
    try: model=load_model('add_youknow',os.path.join(os.path.dirname(sys.path[0]),'pre_trained_models/add_youknow.pth'),dev)
    except Exception as e: return {"error":f"Tải model thất bại: {e}"}
    os.makedirs(outdir,exist_ok=True)
    files=[f for f in os.listdir(indir) if f.lower().endswith(('.png','.jpg'))]
    for fn in tqdm(files,file=sys.stderr):
        try:
            t,s=load_image_to_tensor(os.path.join(indir,fn));
            with torch.no_grad(): mt=torch.sigmoid(model(t.to(dev)))
            save_tensor_as_image(mt,os.path.join(outdir,fn),s)
        except Exception as e: sys.stderr.write(f"\\nLỗi file '{fn}': {e}\\n")
    return {"processed_frame_folder":os.path.abspath(outdir)}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input_dir',required=True); p.add_argument('--output_dir',required=True); a=p.parse_args();
    print(json.dumps(process(a.input_dir,a.output_dir,'cuda' if torch.cuda.is_available() else 'cpu')))
if __name__=="__main__": main()"""),
    ("script_AI/run/run_mosaic_position.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os,torch; from tqdm import tqdm; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from script_AI.model_loader import load_model; from script_AI.images_processing import load_image_to_tensor,save_tensor_as_image
def process(indir,outdir,dev):
    try: model=load_model('mosaic_position',os.path.join(os.path.dirname(sys.path[0]),'pre_trained_models/mosaic_position.pth'),dev)
    except Exception as e: return {"error":f"Tải model thất bại: {e}"}
    os.makedirs(outdir,exist_ok=True)
    files=[f for f in os.listdir(indir) if f.lower().endswith(('.png','.jpg'))]
    for fn in tqdm(files,file=sys.stderr):
        try:
            t,s=load_image_to_tensor(os.path.join(indir,fn));
            with torch.no_grad(): mt=torch.sigmoid(model(t.to(dev)))
            save_tensor_as_image(mt,os.path.join(outdir,fn),s)
        except Exception as e: sys.stderr.write(f"\\nLỗi file '{fn}': {e}\\n")
    return {"processed_frame_folder":os.path.abspath(outdir)}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input_dir',required=True); p.add_argument('--output_dir',required=True); a=p.parse_args();
    print(json.dumps(process(a.input_dir,a.output_dir,'cuda' if torch.cuda.is_available() else 'cpu')))
if __name__=="__main__": main()"""),
    ("script_AI/run/run_clean_youknow.py", """# -*- coding: utf-8 -*-
import argparse,json,sys,os,torch; from tqdm import tqdm; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from script_AI.model_loader import load_model; from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor,save_tensor_as_image
def process(imgdir,maskdir,outdir,dev):
    try: model=load_model('clean_youknow',os.path.join(os.path.dirname(sys.path[0]),'pre_trained_models/clean_youknow.pth'),dev)
    except Exception as e: return {"error":f"Tải model thất bại: {e}"}
    os.makedirs(outdir,exist_ok=True)
    files=[f for f in os.listdir(imgdir) if f.lower().endswith(('.png','.jpg'))]
    for fn in tqdm(files,file=sys.stderr):
        mp=os.path.join(maskdir,fn)
        if not os.path.exists(mp): continue
        try:
            it,s=load_image_to_tensor(os.path.join(imgdir,fn),normalize=False); it=it*2-1
            mt=load_mask_to_tensor(mp)
            it,mt=it.to(dev),mt.to(dev)
            mi=torch.cat((it,mt),dim=1)
            with torch.no_grad(): inpainted=model(mi,mt)
            final=(inpainted*mt)+(it*(1-mt))
            save_tensor_as_image(final,os.path.join(outdir,fn),s,is_inpainting=True)
        except Exception as e: sys.stderr.write(f"\\nLỗi file '{fn}': {e}\\n")
    return {"processed_frame_folder":os.path.abspath(outdir)}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input_dir',required=True); p.add_argument('--mask_dir',required=True); p.add_argument('--output_dir',required=True); a=p.parse_args();
    print(json.dumps(process(a.input_dir,a.mask_dir,a.output_dir,'cuda' if torch.cuda.is_available() else 'cpu')))
if __name__=="__main__": main()"""),
    ("script_AI/train/train_segmentation.py", """# -*- coding: utf-8 -*-
import torch,os,sys; import torch.nn as nn; import torch.optim as optim; from torch.utils.data import Dataset,DataLoader; from tqdm import tqdm; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor
class SegDS(Dataset):
    def __init__(self,id,md): self.id,self.md,self.f=id,md,sorted(os.listdir(id))
    def __len__(self): return len(self.f)
    def __getitem__(self,i):
        imp,mp=os.path.join(self.id,self.f[i]),os.path.join(self.md,self.f[i])
        img,_=load_image_to_tensor(imp); msk=load_mask_to_tensor(mp)
        return img.squeeze(0),msk.squeeze(0)
def train(m,d,l,o,c):
    m.train()
    for i,k in tqdm(l,file=sys.stderr,desc="Training Epoch"):
        i,k=i.to(d),k.to(d); o.zero_grad(); loss=c(m(i),k); loss.backward(); o.step()
def run_training_session(m,dp,e,lr,sp):
    dev='cuda' if torch.cuda.is_available() else 'cpu'; m.to(dev)
    ds=SegDS(os.path.join(dp,'images'),os.path.join(dp,'masks'))
    l=DataLoader(ds,batch_size=4,shuffle=True,num_workers=2)
    opt=optim.Adam(m.parameters(),lr=lr); crit=nn.BCEWithLogitsLoss()
    for ep in range(e):
        sys.stderr.write(f"\\n--- Epoch {ep+1}/{e} ---\\n")
        train(m,dev,l,opt,crit); torch.save(m.state_dict(),sp)
        sys.stderr.write(f"Đã lưu checkpoint: {sp}\\n")"""),
    ("script_AI/train/train_clean_youknow.py", """# -*- coding: utf-8 -*-
import torch,os,sys; import torch.nn as nn; import torch.optim as optim; from torch.utils.data import Dataset,DataLoader; from tqdm import tqdm; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor
class InpaintDS(Dataset):
    def __init__(self,dp):
        self.gtd=os.path.join(dp,'original_images'); self.md=os.path.join(dp,'mosaiced_images'); self.mskd=os.path.join(dp,'mosaic_masks')
        self.f=sorted(os.listdir(self.gtd))
    def __len__(self):return len(self.f)
    def __getitem__(self,i):
        gt,_=load_image_to_tensor(os.path.join(self.gtd,self.f[i]),normalize=False)
        mo,_=load_image_to_tensor(os.path.join(self.md,self.f[i]),normalize=False)
        msk=load_mask_to_tensor(os.path.join(self.mskd,self.f[i]))
        return gt.squeeze(0),mo.squeeze(0),msk.squeeze(0)
def train(m,d,l,o,c):
    m.train()
    for gt,mo,msk in tqdm(l,file=sys.stderr,desc="Training Epoch"):
        gt,mo,msk=gt.to(d),mo.to(d),msk.to(d)
        mi=torch.cat((mo*2-1,msk),1); o.zero_grad(); out=m(mi,msk); loss=c(out*msk,gt*msk); loss.backward(); o.step()
def run_training_session_inpaint(m,dp,e,lr,sp):
    dev='cuda' if torch.cuda.is_available() else 'cpu'; m.to(dev)
    ds=InpaintDS(dp); l=DataLoader(ds,batch_size=2,shuffle=True,num_workers=2)
    opt=optim.Adam(m.parameters(),lr=lr); crit=nn.L1Loss()
    for ep in range(e):
        sys.stderr.write(f"\\n--- Epoch {ep+1}/{e} ---\\n")
        train(m,dev,l,opt,crit); torch.save(m.state_dict(),sp)
        sys.stderr.write(f"Đã lưu checkpoint: {sp}\\n")"""),
    # --- READMEs ---
    ("README.md", """# Dự án UnOrCensored
Bộ công cụ xử lý video và hình ảnh bằng AI, thiết kế theo dạng module hóa.
## Cài đặt
`python setup.py`
## Sử dụng
- **Xử lý:** `python run.py --file_path <path> --task_name <task>`
- **Huấn luyện:** `python train.py --pre_trained_models_name <model_name>`"""),
    ("data/README.md", "# Thư mục Dữ liệu (data)\\nChứa các bộ dữ liệu để huấn luyện và tinh chỉnh model."),
    ("data/add_youknow/README.md", "# Dữ liệu cho `add_youknow`\\n- `images/`: Ảnh gốc.\\n- `masks/`: Mask tương ứng."),
    ("data/add_youknow/images/README.md", "# Thư mục `images`\\nLưu trữ ảnh gốc cho tác vụ `add_youknow`."),
    ("data/add_youknow/masks/README.md", "# Thư mục `masks`\\nLưu trữ ảnh mask (ground-truth) cho tác vụ `add_youknow`."),
    ("data/clean_youknow/README.md", "# Dữ liệu cho `clean_youknow` (Inpainting)\\n- `original_images/`: Ảnh gốc (ground-truth).\\n- `mosaiced_images/`: Ảnh đã bị che.\\n- `mosaic_masks/`: Vị trí vùng bị che."),
    ("data/clean_youknow/original_images/README.md", "# Thư mục `original_images`\\nLưu trữ ảnh gốc, chưa bị che."),
    ("data/clean_youknow/mosaiced_images/README.md", "# Thư mục `mosaiced_images`\\nLưu trữ ảnh đã bị che, là đầu vào cho model inpainting."),
    ("data/clean_youknow/mosaic_masks/README.md", "# Thư mục `mosaic_masks`\\nLưu trữ mask chỉ định vùng bị che."),
    ("data/mosaic_position/README.md", "# Dữ liệu cho `mosaic_position`\\n- `mosaiced_images/`: Ảnh đã bị che (đầu vào).\\n- `mosaic_masks/`: Mask thực tế (ground-truth)."),
    ("data/mosaic_position/mosaiced_images/README.md", "# Thư mục `mosaiced_images`\\nLưu trữ ảnh đã bị che để model học cách xác định vị trí."),
    ("data/mosaic_position/mosaic_masks/README.md", "# Thư mục `mosaic_masks`\\nLưu trữ mask ground-truth cho `mosaic_position`."),
    ("output/README.md", "# Thư mục `output`\\nNơi lưu trữ kết quả cuối cùng từ `run.py`."),
    ("pre_trained_models/README.md", "# Thư mục `pre_trained_models`\\nChứa các file trọng số `.pth` của model. `setup.py` tải model gốc vào đây, và `train.py` lưu model đã tinh chỉnh vào đây."),
    ("script_AI/README.md", "# Thư mục `script_AI`\\nChứa mã nguồn liên quan đến AI.\\n- `models.py`: Kiến trúc mạng.\\n- `model_loader.py`: Tải model.\\n- `images_processing.py`: Xử lý ảnh.\\n- `run/`: Script thực thi inference.\\n- `train/`: Logic huấn luyện."),
    ("script_AI/run/README.md", "# Thư mục `run`\\nChứa các script thực thi inference, được gọi bởi `run.py`."),
    ("script_AI/train/README.md", "# Thư mục `train`\\nChứa các module logic huấn luyện, được gọi bởi `train.py`."),
    ("tool/README.md", "# Thư mục `tool`\\nChứa các script công cụ xử lý media độc lập, được gọi bởi `run.py`.")
]

def create_project_files(base_dir="UnOrCensored_Project"):
    """
    Tạo toàn bộ cấu trúc thư mục và file cho dự án.
    """
    if os.path.exists(base_dir):
        print(f"Thư mục '{base_dir}' đã tồn tại. Vui lòng xóa hoặc đổi tên và chạy lại.")
        return
        
    os.makedirs(base_dir)
    original_dir = os.getcwd()
    os.chdir(base_dir)

    print(f"Đang tạo dự án trong thư mục: {os.path.abspath(os.getcwd())}")

    for file_path, file_content in ALL_FILES:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        try:
            # Ghi file với encoding utf-8
            with open(file_path, 'w', encoding='utf-8') as f:
                # Ghi nội dung đã được strip() để tránh các dòng trắng không mong muốn
                f.write(file_content.strip())
            print(f"Đã tạo file: {file_path}")
        except IOError as e:
            print(f"Lỗi khi tạo file {file_path}: {e}")

    os.chdir(original_dir)
    print("\\n[Thành công] Toàn bộ cấu trúc dự án đã được tạo trong thư mục '{base_dir}'.")
    print("Các bước tiếp theo:")
    print(f"1. cd {base_dir}")
    print("2. python setup.py")
    print("3. python run.py --help")

if __name__ == "__main__":
    create_project_files()


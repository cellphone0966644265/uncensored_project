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
        """import os, subprocess, sys, gdown
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER_ID = "16qdCbG0P3cAR-m3P2xZ3q6mKY_QFW-i-"
PRE_TRAINED_MODELS_DIR = os.path.join(SCRIPT_DIR, "pre_trained_models")
REQUIREMENTS_FILE = os.path.join(SCRIPT_DIR, "requirements.txt")

def install_requirements():
    print(">>> Bước 1: Bắt đầu cài đặt các thư viện...")
    if not os.path.isfile(REQUIREMENTS_FILE): sys.exit(f">>> [Lỗi] Không tìm thấy file: {REQUIREMENTS_FILE}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE], check=True, capture_output=True, text=True, encoding='utf-8')
        print(">>> [Thành công] Đã cài đặt xong.")
    except subprocess.CalledProcessError as e: sys.exit(f">>> [Lỗi] Cài đặt thất bại:\\n{e.stderr}")

def download_models():
    print(f"\\n>>> Bước 2: Bắt đầu tải models...")
    try:
        gdown.download_folder(id=MODEL_FOLDER_ID, output=PRE_TRAINED_MODELS_DIR, quiet=False, use_cookies=False)
        print(">>> [Thành công] Đã tải xong các model.")
    except Exception as e: sys.exit(f">>> [Lỗi] Tải model thất bại: {e}")

def create_project_structure():
    print("\\n>>> Bước 3: Bắt đầu tạo cấu trúc thư mục...")
    dirs = ["data/add_youknow/images","data/add_youknow/masks","data/mosaic_position/mosaiced_images","data/mosaic_position/mosaic_masks","data/clean_youknow/original_images","data/clean_youknow/mosaiced_images","data/clean_youknow/mosaic_masks","output","tmp","pre_trained_models","tool","script_AI", "script_AI/run","script_AI/train"]
    for path in dirs: os.makedirs(os.path.join(SCRIPT_DIR, path), exist_ok=True)
    print(">>> [Thành công] Cấu trúc thư mục đã sẵn sàng.")

def main():
    print("="*60); print(" BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG DỰ ÁN"); print("="*60)
    create_project_structure(); install_requirements(); download_models()
    print("\\n" + "="*60); print(" HOÀN TẤT! Môi trường đã được chuẩn bị."); print("="*60)

if __name__ == "__main__":
    main()"""
    ),
    (
        "run.py",
        """import argparse, json, sys, os, subprocess, shutil, datetime
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.join(SCRIPT_DIR, "tool")
SCRIPT_AI_RUN_DIR = os.path.join(SCRIPT_DIR, "script_AI", "run")
TMP_DIR = os.path.join(SCRIPT_DIR, "tmp")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

def run_command(command):
    sys.stderr.write(f"\\n--- EXECUTE: {' '.join(command)} ---\\n")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', cwd=SCRIPT_DIR)
        if process.stderr: sys.stderr.write(process.stderr)
        return json.loads(process.stdout)
    except subprocess.CalledProcessError as e: sys.exit(f"[Lỗi Script] Lệnh thất bại.\\nLỗi:\\n{e.stderr}")
    except Exception as e: sys.exit(f"[Lỗi Script] Lỗi không xác định: {e}")

def handle_image(args, temp_dir):
    input_dir = os.path.join(temp_dir, "input"); output_dir = os.path.join(temp_dir, "output")
    os.makedirs(input_dir, exist_ok=True); os.makedirs(output_dir, exist_ok=True)
    shutil.copy(args.file_path, input_dir)
    
    if args.task_name == 'add_youknow':
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_add_youknow.py"), "--input_dir", input_dir, "--output_dir", output_dir])
    elif args.task_name == 'clean_youknow':
        mask_dir = os.path.join(temp_dir, "generated_masks")
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_mosaic_position.py"), "--input_dir", input_dir, "--output_dir", mask_dir])
        run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_clean_youknow.py"), "--input_dir", input_dir, "--mask_dir", mask_dir, "--output_dir", output_dir])
    
    final_dest_dir = args.folder_path or OUTPUT_DIR; os.makedirs(final_dest_dir, exist_ok=True)
    final_dest_file = os.path.join(final_dest_dir, os.path.basename(args.file_path))
    shutil.move(os.path.join(output_dir, os.path.basename(args.file_path)), final_dest_file)
    sys.stderr.write(f"Đã lưu kết quả vào: {final_dest_file}\\n")

def handle_video(args, temp_dir):
    info = run_command(["python", os.path.join(TOOL_DIR, "get_file_info.py"), "--input", args.file_path])
    metadata_json = json.dumps(info.get("metadata", {}))
    duration = run_command(["python", os.path.join(TOOL_DIR, "duration_split.py"), "--input", args.file_path, "--fps", str(info['metadata']['fps'])]).get("optimal_chunk_duration", 300)
    chunks_dir = os.path.join(temp_dir, "chunks"); chunk_paths = run_command(["python", os.path.join(TOOL_DIR, "split_video.py"), "--input", args.file_path, "--duration", str(duration), "--output_dir", chunks_dir]).get("chunk_paths", [])
    
    processed_chunks = []
    for i, chunk_path in enumerate(chunk_paths):
        chunk_proc_dir = os.path.join(temp_dir, f"chunk_{i}")
        frames_info = run_command(["python", os.path.join(TOOL_DIR, "video_to_frames.py"), "--input", chunk_path, "--output_dir", chunk_proc_dir])
        if args.task_name == 'add_youknow':
            processed_frames_dir = os.path.join(chunk_proc_dir, "processed")
            run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_add_youknow.py"), "--input_dir", frames_info['frame_folder'], "--output_dir", processed_frames_dir])
        elif args.task_name == 'clean_youknow':
            mask_dir = os.path.join(chunk_proc_dir, "masks")
            processed_frames_dir = os.path.join(chunk_proc_dir, "processed")
            run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_mosaic_position.py"), "--input_dir", frames_info['frame_folder'], "--output_dir", mask_dir])
            run_command(["python", os.path.join(SCRIPT_AI_RUN_DIR, "run_clean_youknow.py"), "--input_dir", frames_info['frame_folder'], "--mask_dir", mask_dir, "--output_dir", processed_frames_dir])
        proc_chunk_cmd = ["python", os.path.join(TOOL_DIR, "frames_to_video.py"), "--frame_folder", processed_frames_dir, "--metadata_json", metadata_json, "--output", os.path.join(chunk_proc_dir, "out.mp4")]
        if frames_info.get('audio_path'): proc_chunk_cmd.extend(["--audio_path", frames_info['audio_path']])
        processed_chunks.append(run_command(proc_chunk_cmd).get("output_path"))
    
    list_file = os.path.join(temp_dir, "mergelist.txt")
    with open(list_file, "w", encoding='utf-8') as f:
        for p in processed_chunks:
            clean_path = p.replace('\\\\', '/')
            f.write(f"file '{clean_path}'\\n")
            
    final_path = os.path.join(args.folder_path or OUTPUT_DIR, f"final_{os.path.basename(args.file_path)}")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    run_command(["python", os.path.join(TOOL_DIR, "merge_video.py"), "--input_list_file", list_file, "--output", final_path])

def main():
    parser = argparse.ArgumentParser(description="Script điều phối chính cho dự án UnOrCensored.")
    parser.add_argument('--file_path', required=True); parser.add_argument('--task_name', required=True, choices=['add_youknow', 'clean_youknow']); parser.add_argument('--folder_path')
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.append(SCRIPT_DIR)
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
    ("script_AI/__init__.py", "# This file makes the 'script_AI' directory a Python package."),
    ("script_AI/train/__init__.py", "# This file makes the 'train' directory a Python package."),
    (
        "script_AI/model_loader.py",
        """import torch, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from script_AI.models import AddYouknowModel, CleanYouknowModel
def load_model(name, pth, dev='cpu'):
    if name in ['add_youknow', 'mosaic_position']: model = AddYouknowModel()
    elif name == 'clean_youknow': model = CleanYouknowModel()
    else: raise ValueError(f"Model không hỗ trợ: {name}")
    state_dict = torch.load(pth, map_location=torch.device(dev))
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    corrected_dict = {k.replace('saptial_path', 'spatial_path'): v for k, v in state_dict.items()}
    model.load_state_dict(corrected_dict, strict=False)
    return model.to(dev).eval()"""
    ),
    (
        "script_AI/models.py",
        """import torch, torch.nn as nn, torch.nn.functional as F
class ConvBlock(nn.Module):
    def __init__(self, i, o, k, s, p): super().__init__(); self.c=nn.Conv2d(i,o,k,s,p,bias=False); self.b=nn.BatchNorm2d(o); self.r=nn.ReLU(True)
    def forward(self, x): return self.r(self.b(self.c(x)))
class AttentionRefinementModule(nn.Module):
    def __init__(self, i, o): super().__init__(); self.c=ConvBlock(i,o,3,1,1); self.a=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(o,o,1,bias=False),nn.BatchNorm2d(o),nn.Sigmoid())
    def forward(self, x): f=self.c(x); return f*self.a(f)
class FeatureFusionModule(nn.Module):
    def __init__(self, i, o, **kwargs):
        super().__init__(); self.cb=ConvBlock(i,o,1,1,0); self.c1=nn.Conv2d(o,o//4,1); self.r=nn.ReLU(True); self.c2=nn.Conv2d(o//4,o,1); self.sig=nn.Sigmoid()
    def forward(self, fsp, fcp):
        f=self.cb(torch.cat([fsp,fcp],1)); a=F.adaptive_avg_pool2d(f,1); a=self.r(self.c1(a)); a=self.sig(self.c2(a)); return f + f*a
class ContextPath(nn.Module):
    def __init__(self):
        super().__init__(); resnet=torch.hub.load('pytorch/vision:v0.10.0','resnet18',pretrained=True,verbose=False)
        self.conv1=resnet.conv1; self.bn1=resnet.bn1; self.relu=resnet.relu; self.maxpool=resnet.maxpool
        self.layer1=resnet.layer1; self.layer2=resnet.layer2; self.layer3=resnet.layer3; self.layer4=resnet.layer4
    def forward(self, x):
        x=self.relu(self.bn1(self.conv1(x))); x=self.maxpool(x); x=self.layer1(x); x=self.layer2(x); f8=self.layer3(x); f16=self.layer4(f8); return f8, f16
class AddYouknowModel(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__(); self.context_path=ContextPath(); self.spatial_path=nn.Sequential(ConvBlock(3,64,7,2,3),ConvBlock(64,64,3,2,1),ConvBlock(64,128,1,1,0))
        self.attention_refinement_module1=AttentionRefinementModule(256,128); self.attention_refinement_module2=AttentionRefinementModule(512,128)
        self.supervision1=nn.Conv2d(128,n_classes,1); self.supervision2=nn.Conv2d(128,n_classes,1)
        self.feature_fusion_module=FeatureFusionModule(256,n_classes); self.conv=nn.Conv2d(n_classes,n_classes,1)
    def forward(self, x):
        H,W=x.size()[2:]; sp_out=self.spatial_path(x); feat8,feat16=self.context_path(x); g_avg=F.adaptive_avg_pool2d(feat16,1)
        arm2_out=self.attention_refinement_module2(feat16); arm2_out=arm2_out+g_avg; arm2_up=F.interpolate(arm2_out,size=feat8.shape[2:],mode='bilinear',align_corners=False)
        arm1_out=self.attention_refinement_module1(feat8); fuse_out=arm1_out+arm2_up
        fuse_up=F.interpolate(fuse_out,size=sp_out.shape[2:],mode='bilinear',align_corners=False)
        final_out=self.feature_fusion_module(sp_out,fuse_up)
        return self.conv(final_out)
class PConv(nn.Module):
    def __init__(self,i,o,bn=True,sample='none',activ='relu',bias=False):
        super().__init__(); self.sampler=None
        if sample=='down':self.sampler=nn.MaxPool2d(2)
        elif sample=='up':self.sampler=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv=nn.Conv2d(i,o,3,1,1,bias=bias); self.mask_conv=nn.Conv2d(i,o,3,1,1,bias=False); torch.nn.init.constant_(self.mask_conv.weight,1.0)
        [p.requires_grad_(False) for p in self.mask_conv.parameters()]
        self.bn=nn.BatchNorm2d(o) if bn else None; self.activ=nn.ReLU(True) if activ=='relu' else nn.LeakyReLU(0.2,True)
    def forward(self,x,m):
        if self.sampler:x,m=self.sampler(x),self.sampler(m)
        with torch.no_grad():mr=1/(self.mask_conv(m)+1e-8)
        o=self.conv(x*m)*mr;
        if self.bn:o=self.bn(o)
        return self.activ(o),m
class CleanYouknowModel(nn.Module):
    def __init__(self):
        super().__init__(); self.e1=PConv(4,64,bn=False,sample='down'); self.e2=PConv(64,128,sample='down'); self.e3=PConv(128,256,sample='down'); self.e4=PConv(256,512,sample='down');
        self.d1=PConv(512+256,256,activ='leaky',sample='up'); self.d2=PConv(256+128,128,activ='leaky',sample='up');
        self.d3=PConv(128+64,64,activ='leaky',sample='up'); self.d4=PConv(64+4,3,bn=False,activ='leaky'); self.final=nn.Conv2d(3,3,1)
    def forward(self,x,m):
        e1,m1=self.e1(x,m);e2,m2=self.e2(e1,m1);e3,m3=self.e3(e2,m2);e4,m4=self.e4(e3,m3); d1,_=self.d1(torch.cat([e4,e3],1),torch.cat([m4,m3],1));d2,_=self.d2(torch.cat([d1,e2],1),torch.cat([m2,m2],1));d3,_=self.d3(torch.cat([d2,e1],1),torch.cat([m2,m1],1));d4,_=self.d4(torch.cat([d3,x],1),torch.cat([m1,mask],1)); return torch.tanh(self.final(d4))"""
    ),
    (
        "script_AI/run/run_add_youknow.py",
        """import argparse,json,sys,os,torch; from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script_AI.model_loader import load_model
from script_AI.images_processing import load_image_to_tensor,save_tensor_as_image
def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--project_root',required=True);a=p.parse_args();
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(a.project_root, 'pre_trained_models/add_youknow.pth')
    model=load_model('add_youknow', model_path, dev)
    os.makedirs(a.output_dir,exist_ok=True)
    for fn in tqdm(os.listdir(a.input_dir),file=sys.stderr):
        try:
            t,s=load_image_to_tensor(os.path.join(a.input_dir,fn))
            with torch.no_grad(): mt=torch.sigmoid(model(t.to(dev)))
            save_tensor_as_image(mt,os.path.join(a.output_dir,fn),s)
        except Exception as e: sys.stderr.write(f"Lỗi file {fn}: {e}\\n")
    print(json.dumps({'processed_frame_folder':os.path.abspath(a.output_dir)}))
if __name__=="__main__":main()"""
    ),
    (
        "script_AI/run/run_mosaic_position.py",
        """import argparse,json,sys,os,torch; from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script_AI.model_loader import load_model
from script_AI.images_processing import load_image_to_tensor,save_tensor_as_image
def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--project_root',required=True);a=p.parse_args();
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(a.project_root, 'pre_trained_models/mosaic_position.pth')
    model=load_model('mosaic_position', model_path, dev)
    os.makedirs(a.output_dir,exist_ok=True)
    for fn in tqdm(os.listdir(a.input_dir),file=sys.stderr):
        try:
            t,s=load_image_to_tensor(os.path.join(a.input_dir,fn))
            with torch.no_grad(): mt=torch.sigmoid(model(t.to(dev)))
            save_tensor_as_image(mt,os.path.join(a.output_dir,fn),s)
        except Exception as e: sys.stderr.write(f"Lỗi file {fn}: {e}\\n")
    print(json.dumps({'processed_frame_folder':os.path.abspath(a.output_dir)}))
if __name__=="__main__":main()"""
    ),
    (
        "script_AI/run/run_clean_youknow.py",
        """import argparse,json,sys,os,torch; from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script_AI.model_loader import load_model
from script_AI.images_processing import load_image_to_tensor,load_mask_to_tensor,save_tensor_as_image
def main():
    p=argparse.ArgumentParser();p.add_argument('--input_dir',required=True);p.add_argument('--mask_dir',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--project_root',required=True);a=p.parse_args();
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(a.project_root, 'pre_trained_models/clean_youknow.pth')
    model=load_model('clean_youknow', model_path, dev)
    os.makedirs(a.output_dir,exist_ok=True)
    for fn in tqdm(os.listdir(a.input_dir),file=sys.stderr):
        mp=os.path.join(a.mask_dir,fn);
        if not os.path.exists(mp): continue
        try:
            it,s=load_image_to_tensor(os.path.join(a.input_dir,fn),normalize=False);it_norm=it*2-1
            mt=load_mask_to_tensor(mp);it_norm,mt=it_norm.to(dev),mt.to(dev)
            with torch.no_grad():
                out=model(torch.cat((it_norm,mt),dim=1),mt)
                final=(out*mt)+(it_norm*(1-mt))
            save_tensor_as_image(final,os.path.join(a.output_dir,fn),s,is_inpaint=True)
        except Exception as e: sys.stderr.write(f"Lỗi file {fn}: {e}\\n")
    print(json.dumps({'processed_frame_folder':os.path.abspath(a.output_dir)}))
if __name__=="__main__":main()"""
    ),
]

def create_project_files(base_dir="UnOrCensored_Project"):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"Đã xóa thư mục cũ: '{base_dir}'")
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



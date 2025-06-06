# UnOrCensored/script_AI/run/run_clean_youknow.py
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import model_loader
import images_processing
try:
    from . import run_mosaic_position
except ImportError:
    import run_mosaic_position

# (Hàm predict_single_clean_image giữ nguyên)
def predict_single_clean_image(model, image_path, mask_path, device):
    # ...
    # (Nội dung hàm này không thay đổi)
    # ...
    return final_image.astype(np.uint8)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Làm sạch, tái tạo vùng bị che mosaic trên ảnh.")
    parser.add_argument('--input_path', required=True, help="Đường dẫn đến thư mục ảnh bị mosaic.")
    parser.add_argument('--output_path', required=True, help="Đường dẫn để lưu kết quả.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # Tải trước các model cần thiết
    try:
        clean_model = model_loader.load_model('clean_youknow', device=device)
        position_model = model_loader.load_model('mosaic_position', device=device)
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi tải model: {e}")
        # Ném lỗi ra ngoài để run.py biết và dừng lại
        raise e

    os.makedirs(args.output_path, exist_ok=True)
    image_files = sorted([f for f in os.listdir(args.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Tạo thư mục tạm để lưu mask
    tmp_mask_dir = os.path.join(os.path.dirname(args.output_path), 'tmp_masks_for_cleaning')
    if os.path.exists(tmp_mask_dir): shutil.rmtree(tmp_mask_dir)
    os.makedirs(tmp_mask_dir, exist_ok=True)
    
    print("Bước 1: Phát hiện vị trí mosaic...")
    # Gọi trực tiếp hàm predict thay vì cả script để kiểm soát tốt hơn
    for filename in tqdm(image_files, desc="Finding mosaic positions"):
        input_image_path = os.path.join(args.input_path, filename)
        output_mask_path = os.path.join(tmp_mask_dir, filename)
        mask = run_mosaic_position.predict_single_image(position_model, input_image_path, device)
        images_processing.save_image(mask, output_mask_path)
    print("✅ Đã tạo xong mask vị trí mosaic.")

    print("\nBước 2: Bắt đầu làm sạch (inpainting)...")
    for filename in tqdm(image_files, desc="Cleaning frames"):
        input_image_path = os.path.join(args.input_path, filename)
        mask_path = os.path.join(tmp_mask_dir, filename)
        output_image_path = os.path.join(args.output_path, filename)
        
        # SỬA LỖI LOGIC: Dừng lại nếu không có mask, không chép file gốc
        if not os.path.exists(mask_path) or os.path.getsize(mask_path) == 0:
            print(f"❌ LỖI: Không thể tạo hoặc không tìm thấy mask cho {filename}. Dừng xử lý file này.")
            # Thay vì chép file gốc, ta có thể bỏ qua hoặc tạo 1 ảnh đen để báo lỗi
            # Ở đây ta bỏ qua
            continue

        cleaned_image = predict_single_clean_image(clean_model, input_image_path, mask_path, device)
        images_processing.save_image(cleaned_image, output_image_path)
    
    shutil.rmtree(tmp_mask_dir)

if __name__ == '__main__':
    if len(sys.argv) == 1: sys.argv.extend(['-h'])
    main()


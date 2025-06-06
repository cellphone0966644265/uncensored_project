# UnOrCensored/script_AI/run/run_clean_youknow.py
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Thêm đường dẫn gốc của script_AI vào sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import model_loader
import images_processing

# Import script `run_mosaic_position` để có thể sử dụng nó lấy mask
# Điều này thể hiện sự liên kết giữa các module như trong tài liệu
try:
   from . import run_mosaic_position
except ImportError:
   import run_mosaic_position


def predict_single_image(model, image_path, mask_path, device):
   """
   Tái tạo lại vùng bị che mờ trên ảnh.

   Args:
       model (torch.nn.Module): Model InpaintingGenerator đã được tải.
       image_path (str): Đường dẫn đến ảnh bị che mờ.
       mask_path (str): Đường dẫn đến ảnh mask của vùng bị che mờ.
       device (torch.device): Thiết bị để chạy model.

   Returns:
       np.ndarray: Ảnh đã được tái tạo.
   """
   # Định nghĩa các bước tiền xử lý
   to_tensor = transforms.ToTensor()

   # Tải ảnh và mask
   img = Image.open(image_path).convert('RGB')
   mask = Image.open(mask_path).convert('L') # Load as grayscale
   
   # Đảm bảo mask là nhị phân (0 hoặc 1)
   mask = mask.point(lambda p: p > 128 and 255)
   
   # Chuyển sang tensor
   img_tensor = to_tensor(img)
   mask_tensor = to_tensor(mask)

   # Chuẩn hóa tensor ảnh trong khoảng [-1, 1] như yêu cầu của nhiều model inpainting
   img_tensor = (img_tensor * 2.0) - 1.0
   
   # Nối ảnh bị che và mask lại, thêm batch dimension
   input_tensor = torch.cat([img_tensor, mask_tensor], dim=0).unsqueeze(0).to(device)

   # Chạy dự đoán
   with torch.no_grad():
       output_tensor = model(input_tensor[:, :3, :, :], input_tensor[:, 3:, :, :])
   
   # Post-process output
   # Chuyển output về khoảng [0, 1] rồi [0, 255]
   output_tensor = (output_tensor.squeeze(0).cpu().detach() + 1.0) / 2.0
   output_image = output_tensor.mul(255).permute(1, 2, 0).byte().numpy()

   # Ghép ảnh gốc và ảnh đã tái tạo
   original_np = np.array(img)
   mask_np = np.array(mask.convert('RGB')) / 255
   
   final_image = original_np * (1 - mask_np) + output_image * mask_np
   
   return final_image.astype(np.uint8)

def main(argv=None):
   parser = argparse.ArgumentParser(description="Làm sạch, tái tạo vùng bị che mosaic trên ảnh.")
   parser.add_argument('--input_path', required=True, help="Đường dẫn đến ảnh/thư mục ảnh bị mosaic.")
   parser.add_argument('--output_path', required=True, help="Đường dẫn để lưu kết quả.")
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Sử dụng thiết bị: {device}")

   # Tải model clean_youknow
   try:
       clean_model = model_loader.load_model('clean_youknow', device=device)
   except Exception as e:
       print(f"Lỗi khi tải model 'clean_youknow': {e}")
       return
       
   # Xử lý input là thư mục hay file đơn lẻ
   if os.path.isdir(args.input_path):
       os.makedirs(args.output_path, exist_ok=True)
       image_files = sorted([f for f in os.listdir(args.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
       
       # Tạo thư mục tạm để lưu mask
       tmp_mask_dir = os.path.join(os.path.dirname(args.output_path), 'tmp_masks_for_cleaning')
       os.makedirs(tmp_mask_dir, exist_ok=True)
       
       print("Bước 1: Phát hiện vị trí mosaic...")
       # Sử dụng run_mosaic_position để tạo mask cho tất cả các ảnh
       # Đây là một ví dụ về pipeline chaining
       run_mosaic_position.main([
           '--input_path', args.input_path,
           '--output_path', tmp_mask_dir
       ])

       print("Bước 2: Bắt đầu làm sạch (inpainting)...")
       for filename in tqdm(image_files, desc="Cleaning frames"):
           input_image_path = os.path.join(args.input_path, filename)
           mask_path = os.path.join(tmp_mask_dir, filename) # Giả sử tên file mask giống tên file gốc
           output_image_path = os.path.join(args.output_path, filename)
           
           if not os.path.exists(mask_path):
               print(f"Cảnh báo: Không tìm thấy mask cho {filename}. Bỏ qua.")
               # Sao chép ảnh gốc nếu không có mask
               images_processing.save_image(images_processing.load_image(input_image_path), output_image_path)
               continue

           cleaned_image = predict_single_image(clean_model, input_image_path, mask_path, device)
           images_processing.save_image(cleaned_image, output_image_path)
       
       # Dọn dẹp thư mục mask tạm
       import shutil
       shutil.rmtree(tmp_mask_dir)

   elif os.path.isfile(args.input_path):
       # Tạo mask tạm
       tmp_mask_path = os.path.join(os.path.dirname(args.output_path), 'tmp_mask.png')
       
       print("Bước 1: Phát hiện vị trí mosaic...")
       run_mosaic_position.main([
           '--input_path', args.input_path,
           '--output_path', tmp_mask_path
       ])

       if not os.path.exists(tmp_mask_path):
           print("Lỗi: Không thể tạo mask vị trí mosaic.")
           return

       print("Bước 2: Bắt đầu làm sạch (inpainting)...")
       cleaned_image = predict_single_image(clean_model, args.input_path, tmp_mask_path, device)
       images_processing.save_image(cleaned_image, args.output_path)
       print(f"Đã lưu kết quả tại: {args.output_path}")
       
       # Dọn dẹp mask tạm
       os.remove(tmp_mask_path)
   else:
       print(f"Lỗi: Đường dẫn không tồn tại: {args.input_path}")

if __name__ == '__main__':
   if len(sys.argv) == 1:
       sys.argv.extend(['-h'])
   main()
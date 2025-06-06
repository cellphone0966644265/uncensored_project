# UnOrCensored/script_AI/run/run_mosaic_position.py
# Về cơ bản, script này rất giống với run_add_youknow.py
# vì chúng dùng chung kiến trúc model BiSeNet, chỉ khác nhau
# về dữ liệu được huấn luyện (một cái học từ ảnh gốc, một cái học từ ảnh mosaic)
# và trọng số đã được fine-tune.

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Thêm đường dẫn gốc của script_AI
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import model_loader
import images_processing

def predict_single_image(model, image_path, device):
   """
   Dự đoán mask cho một ảnh duy nhất.
   (Hàm này được sao chép từ run_add_youknow vì logic y hệt)
   """
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
   ])
   img = Image.open(image_path).convert('RGB')
   image_tensor = transform(img).unsqueeze(0).to(device)
   with torch.no_grad():
       outputs = model(image_tensor)
   pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
   pred[pred > 0] = 255
   return pred

def main(argv=None):
   parser = argparse.ArgumentParser(description="Phát hiện vị trí mosaic trên ảnh và tạo mask.")
   parser.add_argument('--input_path', required=True, help="Đường dẫn đến ảnh/thư mục ảnh bị mosaic.")
   parser.add_argument('--output_path', required=True, help="Đường dẫn để lưu ảnh/thư mục mask kết quả.")
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Sử dụng thiết bị: {device}")

   try:
       # Tải model 'mosaic_position'
       model = model_loader.load_model('mosaic_position', device=device)
   except Exception as e:
       print(f"Lỗi khi tải model 'mosaic_position': {e}")
       return

   if os.path.isdir(args.input_path):
       os.makedirs(args.output_path, exist_ok=True)
       image_files = sorted([f for f in os.listdir(args.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
       
       print(f"Phát hiện {len(image_files)} ảnh. Bắt đầu tạo mask vị trí mosaic...")
       for filename in tqdm(image_files, desc="Finding mosaic positions"):
           input_image_path = os.path.join(args.input_path, filename)
           output_mask_path = os.path.join(args.output_path, filename) # Lưu với tên gốc
           mask = predict_single_image(model, input_image_path, device)
           images_processing.save_image(mask, output_mask_path)
           
   elif os.path.isfile(args.input_path):
       print("Xử lý file ảnh đơn lẻ...")
       mask = predict_single_image(model, args.input_path, device)
       images_processing.save_image(mask, args.output_path)
       print(f"Đã lưu mask kết quả tại: {args.output_path}")
   else:
       print(f"Lỗi: Đường dẫn không tồn tại: {args.input_path}")

if __name__ == '__main__':
   if len(sys.argv) == 1:
       sys.argv.extend(['-h'])
   main()
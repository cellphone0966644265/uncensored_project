# UnOrCensored/script_AI/run/run_add_youknow.py
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Thêm đường dẫn gốc của script_AI vào sys.path
# để có thể import các module anh em (sibling modules)
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

# Import các module cần thiết từ script_AI
import model_loader
import images_processing

def predict_single_image(model, image_path, device):
   """
   Dự đoán mask cho một ảnh duy nhất.

   Args:
       model (torch.nn.Module): Model BiSeNet đã được tải.
       image_path (str): Đường dẫn đến ảnh đầu vào.
       device (torch.device): Thiết bị để chạy model.

   Returns:
       np.ndarray: Mask dự đoán dưới dạng ảnh thang độ xám (0-255).
   """
   # Định nghĩa các bước tiền xử lý ảnh
   # Phải giống với các bước đã dùng khi huấn luyện model
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
   ])

   # Tải ảnh và tiền xử lý
   img = Image.open(image_path).convert('RGB')
   image_tensor = transform(img).unsqueeze(0).to(device)

   # Chạy dự đoán
   with torch.no_grad():
       outputs = model(image_tensor)
   
   # Post-process output
   # Lấy class có xác suất cao nhất tại mỗi pixel
   pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
   
   # Chuyển mask từ (0, 1, ...) thành (0, 255, ...)
   pred[pred > 0] = 255
   
   return pred

def main(argv=None):
   parser = argparse.ArgumentParser(
       description="Phát hiện đối tượng và tạo mask. "
                   "Kết quả là ảnh mask hoặc ảnh đã được làm mờ trực tiếp."
   )
   parser.add_argument('--input_path', required=True, help="Đường dẫn đến file ảnh hoặc thư mục chứa các frame.")
   parser.add_argument('--output_path', required=True, help="Đường dẫn để lưu file ảnh hoặc thư mục chứa các frame kết quả.")
   parser.add_argument(
       '--output_type', choices=['mask', 'mosaiced'], default='mosaiced',
       help="Loại output: 'mask' chỉ lưu ảnh mask, 'mosaiced' lưu ảnh đã được áp mosaic."
   )
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   # Xác định thiết bị
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Sử dụng thiết bị: {device}")

   # Tải model
   try:
       model = model_loader.load_model('add_youknow', device=device)
   except Exception as e:
       print(f"Lỗi khi tải model: {e}")
       return

   # Xử lý input là thư mục hay file đơn lẻ
   if os.path.isdir(args.input_path):
       os.makedirs(args.output_path, exist_ok=True)
       image_files = sorted([f for f in os.listdir(args.input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
       
       print(f"Phát hiện {len(image_files)} ảnh trong thư mục. Bắt đầu xử lý...")
       for filename in tqdm(image_files, desc="Processing frames"):
           input_image_path = os.path.join(args.input_path, filename)
           output_image_path = os.path.join(args.output_path, filename)

           mask = predict_single_image(model, input_image_path, device)

           if args.output_type == 'mask':
               images_processing.save_image(mask, output_image_path)
           else: # Mosaiced
               original_image = images_processing.load_image(input_image_path)
               mosaiced_image = images_processing.apply_mosaic_from_mask(original_image, mask)
               images_processing.save_image(mosaiced_image, output_image_path)
               
   elif os.path.isfile(args.input_path):
       print("Xử lý file ảnh đơn lẻ...")
       mask = predict_single_image(model, args.input_path, device)

       if args.output_type == 'mask':
           images_processing.save_image(mask, args.output_path)
       else: # Mosaiced
           original_image = images_processing.load_image(args.input_path)
           mosaiced_image = images_processing.apply_mosaic_from_mask(original_image, mask)
           images_processing.save_image(mosaiced_image, args.output_path)
       print(f"Đã lưu kết quả tại: {args.output_path}")
       
   else:
       print(f"Lỗi: Đường dẫn không tồn tại: {args.input_path}")

if __name__ == '__main__':
   # Cho phép chạy script độc lập để test
   # Ví dụ: python script_AI/run/run_add_youknow.py --input_path path/to/image.jpg --output_path path/to/result.jpg
   if len(sys.argv) == 1:
       sys.argv.extend(['-h'])
   main()
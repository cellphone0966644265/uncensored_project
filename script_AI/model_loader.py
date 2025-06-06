# UnOrCensored/script_AI/model_loader.py
import os
import torch
import yaml

# Import các lớp model từ file models.py
try:
   from . import models
except ImportError:
   import models

def load_model(model_name, device='cpu'):
   """
   Tải kiến trúc và trọng số của model dựa trên tên.
   Đây là cơ chế cốt lõi để làm việc với các model AI.
   
   Args:
       model_name (str): Tên của model (ví dụ: 'add_youknow').
       device (str or torch.device): Thiết bị để tải model lên ('cpu' hoặc 'cuda').
       
   Returns:
       torch.nn.Module: Model đã được tải và sẵn sàng để sử dụng.
   """
   project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   models_dir = os.path.join(project_root, 'pre_trained_models')
   
   pth_path = os.path.join(models_dir, f"{model_name}.pth")
   yaml_path = os.path.join(models_dir, f"{model_name}_structure.yaml")

   # --- Kiểm tra sự tồn tại của các file ---
   if not os.path.exists(pth_path):
       raise FileNotFoundError(f"Không tìm thấy file trọng số model: {pth_path}")
   if not os.path.exists(yaml_path):
       print(f"Cảnh báo: Không tìm thấy file cấu trúc {yaml_path}. Vẫn tiếp tục tải model.")

   # --- Dựng lại kiến trúc model dựa vào tên ---
   print(f"Bắt đầu dựng kiến trúc cho model '{model_name}'...")
   model = None
   if model_name in ['add_youknow', 'mosaic_position']:
       # Cả hai model này đều dùng BiSeNet nhưng cho 2 tác vụ khác nhau
       # Giả sử output là một mask đơn (1 class)
       model = models.BiSeNet(n_classes=1)
       print("Đã khởi tạo kiến trúc BiSeNet.")
   elif model_name == 'clean_youknow':
       # Model này dùng kiến trúc Inpainting
       # Input là ảnh + mask (4 channels), output là ảnh (3 channels)
       model = models.InpaintingGenerator(in_channels=4, out_channels=3)
       print("Đã khởi tạo kiến trúc InpaintingGenerator.")
   else:
       raise ValueError(f"Tên model không hợp lệ: '{model_name}'.")

   # --- Nạp trọng số từ file .pth ---
   try:
       print(f"Đang nạp trọng số từ: {pth_path}")
       # Tải state_dict. map_location đảm bảo model được tải đúng thiết bị
       state_dict = torch.load(pth_path, map_location=torch.device(device))
       
       # Một số model được lưu với tiền tố 'module.' nếu huấn luyện với DataParallel
       # Cần xử lý trường hợp này
       if list(state_dict.keys())[0].startswith('module.'):
           # Tạo state_dict mới không có tiền tố 'module.'
           from collections import OrderedDict
           new_state_dict = OrderedDict()
           for k, v in state_dict.items():
               name = k[7:] # bỏ 'module.'
               new_state_dict[name] = v
           state_dict = new_state_dict

       model.load_state_dict(state_dict)
       print("Nạp trọng số thành công.")
       
   except Exception as e:
       print(f"Lỗi nghiêm trọng khi nạp trọng số model: {e}")
       print("Hãy chắc chắn rằng kiến trúc model trong 'models.py' khớp với file trọng số.")
       raise e
       
   # Chuyển model sang thiết bị đã chọn và đặt ở chế độ đánh giá
   model.to(device)
   model.eval()
   
   print(f"Model '{model_name}' đã sẵn sàng trên thiết bị '{device}'.")
   return model

# Có thể thêm hàm main để test
if __name__ == '__main__':
   # Để chạy test, bạn cần có các file model trong thư mục pre_trained_models
   # Ví dụ: python -m script_AI.model_loader
   try:
       print("\n--- Thử tải model 'add_youknow' ---")
       model_add = load_model('add_youknow')
       print(model_add)
   except Exception as e:
       print(f"Không thể tải 'add_youknow': {e}")
       
   try:
       print("\n--- Thử tải model 'clean_youknow' ---")
       model_clean = load_model('clean_youknow')
       print(model_clean)
   except Exception as e:
       print(f"Không thể tải 'clean_youknow': {e}")
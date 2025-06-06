# UnOrCensored/script_AI/images_processing.py
import cv2
import numpy as np
import os
from PIL import Image

def load_image(image_path, to_rgb=True):
   """
   Tải một hình ảnh từ đường dẫn và chuyển đổi sang định dạng numpy array.
   
   Args:
       image_path (str): Đường dẫn đến file ảnh.
       to_rgb (bool): True để chuyển từ BGR (OpenCV) sang RGB.
       
   Returns:
       np.ndarray: Mảng numpy chứa dữ liệu ảnh.
   """
   try:
       # Sử dụng cv2 để đọc vì hiệu năng tốt hơn với các định dạng ảnh phổ biến
       img = cv2.imread(image_path, cv2.IMREAD_COLOR)
       if img is None:
           raise FileNotFoundError(f"Không thể đọc file ảnh: {image_path}")
       if to_rgb:
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       return img
   except Exception as e:
       print(f"Lỗi khi tải ảnh {image_path}: {e}")
       return None

def save_image(image_data, output_path):
   """
   Lưu một mảng numpy ảnh ra file.
   
   Args:
       image_data (np.ndarray): Dữ liệu ảnh (dạng RGB).
       output_path (str): Đường dẫn để lưu file ảnh.
   """
   try:
       # Chuyển từ RGB sang BGR trước khi lưu bằng OpenCV
       img_bgr = cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGB2BGR)
       
       # Tạo thư mục nếu chưa tồn tại
       os.makedirs(os.path.dirname(output_path), exist_ok=True)
       
       cv2.imwrite(output_path, img_bgr)
       return True
   except Exception as e:
       print(f"Lỗi khi lưu ảnh tại {output_path}: {e}")
       return False

def apply_mosaic_from_mask(image, mask, pixel_size=16):
   """
   Áp dụng hiệu ứng mosaic lên một ảnh dựa vào một mask nhị phân.
   
   Args:
       image (np.ndarray): Ảnh gốc (RGB).
       mask (np.ndarray): Ảnh mask trắng đen (255 là vùng cần làm mosaic).
       pixel_size (int): Kích thước của ô mosaic.
       
   Returns:
       np.ndarray: Ảnh đã được áp dụng mosaic.
   """
   if image is None or mask is None:
       return None
       
   img_mosaic = image.copy()
   h, w, _ = img_mosaic.shape

   # Chuyển mask sang dạng nhị phân 0 và 1 để dễ xử lý
   binary_mask = (mask > 128).astype(np.uint8)

   for i in range(0, h, pixel_size):
       for j in range(0, w, pixel_size):
           # Lấy vùng tọa độ
           y_end = min(i + pixel_size, h)
           x_end = min(j + pixel_size, w)
           
           # Kiểm tra xem vùng này có cần làm mosaic không
           # Nếu có ít nhất một pixel trong mask thuộc vùng này, ta sẽ làm mosaic cả khối
           if np.any(binary_mask[i:y_end, j:x_end]):
               # Tính màu trung bình của khối
               block = image[i:y_end, j:x_end]
               if block.size > 0:
                   avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
                   # Gán màu trung bình cho khối đó trong ảnh mosaic
                   img_mosaic[i:y_end, j:x_end] = avg_color
                   
   return img_mosaic
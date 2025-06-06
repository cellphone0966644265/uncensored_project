# UnOrCensored/setup.py
import subprocess
import sys
import os

def install_requirements():
   """
   Cài đặt các thư viện từ file requirements.txt.
   """
   try:
       print("Bắt đầu cài đặt các thư viện cần thiết từ requirements.txt...")
       subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
       print("Cài đặt thư viện thành công.")
       return True
   except subprocess.CalledProcessError as e:
       print(f"Lỗi khi cài đặt thư viện: {e}")
       print("Vui lòng kiểm tra file requirements.txt và thử lại.")
       return False
   except FileNotFoundError:
       print("Lỗi: Không tìm thấy file requirements.txt. Hãy chắc chắn bạn đang chạy script từ thư mục gốc của dự án.")
       return False

def download_models():
   """
   Tải các model đã huấn luyện từ Google Drive bằng gdown.
   """
   try:
       # Import gdown sau khi đã chắc chắn nó được cài đặt
       import gdown
       print("Bắt đầu tải các model... Quá trình này có thể mất một lúc.")
       
       folder_id = "15WVt_ASxEdh3lx16w52h7CfC2dA1QCNO"
       output_path = "pre_trained_models"
       
       if not os.path.exists(output_path):
           os.makedirs(output_path)
           
       # Lệnh gdown để tải toàn bộ thư mục
       gdown.download_folder(id=folder_id, output=output_path, quiet=False, use_cookies=False)
       
       print(f"Tải model thành công và đã lưu vào thư mục '{output_path}'.")

   except ImportError:
       print("Lỗi: không thể import thư viện 'gdown'. Đảm bảo nó đã được cài đặt thành công.")
   except Exception as e:
       print(f"Lỗi không xác định trong quá trình tải model: {e}")
       print("Vui lòng kiểm tra lại Google Drive Folder ID và kết nối mạng.")

if __name__ == "__main__":
   print("--- Bắt đầu quá trình cài đặt cho dự án UnOrCensored ---")
   
   # 1. Cài đặt thư viện
   if install_requirements():
       # 2. Tải model nếu cài thư viện thành công
       download_models()
       
   print("--- Quá trình cài đặt đã hoàn tất ---")
   print("Môi trường đã sẵn sàng để chạy 'run.py' hoặc 'train.py'.")
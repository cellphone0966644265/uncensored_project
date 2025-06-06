# UnOrCensored/setup.py
import subprocess
import sys
import os
import shutil

def install_requirements():
    """
    Cài đặt các thư viện từ file requirements.txt.
    """
    try:
        print("--- Bắt đầu cài đặt các thư viện cần thiết từ requirements.txt... ---")
        # Sử dụng tệp requirements.txt đã được tối giản để tránh xung đột
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Cài đặt thư viện thành công.")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi cài đặt thư viện: {e}")
        return False

def download_models_robust():
    """
    Tải các model từ Google Drive, đảm bảo file được đặt đúng vị trí.
    """
    try:
        import gdown
    except ImportError:
        print("❌ Lỗi: không thể import thư viện 'gdown'. Hãy đảm bảo nó đã được cài đặt.")
        return

    folder_id = "16qdCbG0P3cAR-m3P2xZ3q6mKY_QFW-i-"
    output_path = "pre_trained_models"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"\n--- Bắt đầu tải các model từ Google Drive Folder ID: {folder_id} ---")

    try:
        print("\n... Đang chạy gdown từ thư mục gốc của dự án ...")
        folder_url = f'https://drive.google.com/drive/folders/{folder_id}'
        
        items_before = set(os.listdir('.'))
        
        # Chạy lệnh gdown
        subprocess.run(['gdown', '--folder', folder_url, '-c'], check=True, capture_output=True, text=True)

        items_after = set(os.listdir('.'))
        new_items = items_after - items_before
        
        if not new_items:
            print("\n❌ Lỗi: gdown đã chạy nhưng không có thư mục hoặc file nào mới được tạo.")
            print("Vui lòng kiểm tra lại quyền chia sẻ của thư mục trên Google Drive.")
            return

        downloaded_folder_name = new_items.pop()
        downloaded_folder_path = os.path.join(os.getcwd(), downloaded_folder_name)
        
        if not os.path.isdir(downloaded_folder_path):
            print(f"❌ Lỗi: '{downloaded_folder_name}' được tạo ra không phải là một thư mục.")
            return

        print(f"✅ gdown đã tải dữ liệu vào thư mục tạm: '{downloaded_folder_name}'")
        
        print(f"... Đang di chuyển các file vào '{output_path}' ...")
        all_files = os.listdir(downloaded_folder_path)
        for f in all_files:
            shutil.move(os.path.join(downloaded_folder_path, f), output_path)
            
        os.rmdir(downloaded_folder_path)
        print(f"✅ Đã di chuyển thành công {len(all_files)} file và dọn dẹp thư mục tạm.")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi khi chạy gdown. Lỗi có thể do quyền chia sẻ hoặc ID không hợp lệ.")
        print(f"   Chi tiết lỗi: {e.stderr}")
    except Exception as e:
        print(f"❌ Lỗi không xác định trong quá trình tải model: {e}")

if __name__ == "__main__":
    print("--- Bắt đầu quá trình cài đặt cho dự án UnOrCensored ---")
    
    if install_requirements():
        download_models_robust()
        
    print("\n--- Quá trình cài đặt đã hoàn tất ---")
    print("Môi trường đã sẵn sàng để chạy 'run.py' hoặc 'train.py'.")


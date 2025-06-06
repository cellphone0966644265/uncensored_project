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
        # Sử dụng --no-cache-dir để tránh các vấn đề với cache cũ
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])
        print("✅ Cài đặt thư viện thành công.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi cài đặt thư viện: {e}")
        print("Vui lòng kiểm tra file requirements.txt và thử lại.")
        return False
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file requirements.txt. Hãy chắc chắn bạn đang chạy script từ thư mục gốc của dự án.")
        return False

def download_models_robust():
    """
    Tải các model đã huấn luyện từ Google Drive một cách mạnh mẽ hơn.
    """
    try:
        import gdown
    except ImportError:
        print("❌ Lỗi: không thể import thư viện 'gdown'. Đảm bảo nó đã được cài đặt thành công.")
        return

    # ID thư mục mới được cập nhật từ bạn
    folder_id = "16qdCbG0P3cAR-m3P2xZ3q6mKY_QFW-i-"
    output_path = "pre_trained_models"

    print(f"\n--- Bắt đầu tải các model từ Google Drive Folder ID: {folder_id} ---")
    print(f"Thư mục lưu trữ: '{output_path}'")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    files_before = len(os.listdir(output_path))

    try:
        print("\n... Đang thử phương pháp: Sử dụng gdown từ command line ...")
        folder_url = f'https://drive.google.com/drive/folders/{folder_id}'
        
        # Chuyển tới thư mục output để gdown tải file vào đúng nơi
        current_dir = os.getcwd()
        os.chdir(output_path)
        
        subprocess.run(['gdown', '--folder', folder_url, '-c'], check=True)
        
        os.chdir(current_dir)
        
        files_after = len(os.listdir(output_path))
        if files_after > files_before:
            print(f"✅ Tải thành công {files_after - files_before} file.")
        else:
            print("\n❌ Lỗi: Không tải được file nào.")
            print("Vui lòng kiểm tra lại các yếu tố sau:")
            print(f"  1. ID thư mục có chính xác không: '{folder_id}'.")
            print("  2. Quyền chia sẻ của thư mục trên Google Drive đã được đặt thành 'Anyone with the link' (Bất kỳ ai có đường liên kết) chưa.")

    except Exception as e:
        print(f"❌ Lỗi không xác định trong quá trình tải model: {e}")

if __name__ == "__main__":
    print("--- Bắt đầu quá trình cài đặt cho dự án UnOrCensored ---")
    
    if install_requirements():
        download_models_robust()
        
    print("\n--- Quá trình cài đặt đã hoàn tất ---")
    print("Môi trường đã sẵn sàng để chạy 'run.py' hoặc 'train.py'.")


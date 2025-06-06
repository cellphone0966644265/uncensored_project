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

    # ID thư mục chính xác từ tài liệu PDF (sử dụng chữ 'l' thay vì 'I')
    folder_id = "15WVt_ASxEdh3lx16w52h7CfC2dA1QCNO"
    output_path = "pre_trained_models"

    print(f"\n--- Bắt đầu tải các model từ Google Drive Folder ID: {folder_id} ---")
    print(f"Thư mục lưu trữ: '{output_path}'")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Đếm số file trước khi tải để kiểm tra
    files_before = len(os.listdir(output_path))

    try:
        # Phương pháp 1: Sử dụng hàm gdown.download_folder
        print("\n... Đang thử phương pháp 1: Sử dụng hàm gdown.download_folder ...")
        gdown.download_folder(id=folder_id, output=output_path, quiet=False, use_cookies=False)
        
        # Kiểm tra xem có file nào được tải về không
        files_after = len(os.listdir(output_path))
        if files_after > files_before:
            print(f"✅ Tải thành công {files_after - files_before} file bằng gdown.download_folder.")
            return

        print("🟡 Phương pháp 1 không tải được file nào. Thử phương pháp 2...")

    except Exception as e:
        print(f"🟡 Lỗi với phương pháp 1 (gdown.download_folder): {e}. Thử phương pháp 2...")

    # Phương pháp 2: Sử dụng gdown từ command line qua subprocess (thường ổn định hơn)
    try:
        print("\n... Đang thử phương pháp 2: Sử dụng gdown từ command line ...")
        folder_url = f'https://drive.google.com/drive/folders/{folder_id}'
        
        # Chuyển tới thư mục output để gdown tải file vào đúng nơi
        current_dir = os.getcwd()
        os.chdir(output_path)
        
        # Lệnh gdown để tải toàn bộ thư mục
        subprocess.run(['gdown', '--folder', folder_url, '-c'], check=True)
        
        # Quay lại thư mục ban đầu
        os.chdir(current_dir)
        
        # Kiểm tra lại lần cuối
        files_after = len(os.listdir(output_path))
        if files_after > files_before:
            print(f"✅ Tải thành công {files_after - files_before} file bằng gdown command line.")
        else:
            print("\n❌ Lỗi: Cả hai phương pháp đều không tải được file.")
            print("Vui lòng kiểm tra lại các yếu tố sau:")
            print(f"  1. ID thư mục có chính xác không: '{folder_id}' (Đã sửa thành chữ 'l' thường).")
            print("  2. Quyền chia sẻ của thư mục trên Google Drive đã được đặt thành 'Anyone with the link' (Bất kỳ ai có đường liên kết) chưa.")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Lỗi với phương pháp 2 (gdown command line): {e}")
        print("Vui lòng kiểm tra lại các yếu tố đã nêu ở trên.")
    except Exception as e:
        print(f"❌ Lỗi không xác định trong quá trình tải model: {e}")

if __name__ == "__main__":
    print("--- Bắt đầu quá trình cài đặt cho dự án UnOrCensored ---")
    
    if install_requirements():
        download_models_robust()
        
    print("\n--- Quá trình cài đặt đã hoàn tất ---")
    print("Môi trường đã sẵn sàng để chạy 'run.py' hoặc 'train.py'.")


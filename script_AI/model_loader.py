# UnOrCensored/script_AI/model_loader.py
import os
import torch
import yaml

# Import các lớp model từ file models.py
try:
    from . import models
except ImportError:
    # Cho phép chạy độc lập để test
    import models

def load_model(model_name, device='cpu'):
    """
    Tải kiến trúc và trọng số của model dựa trên tên.
    Sử dụng đường dẫn tương đối để đảm bảo tính linh hoạt.
    
    Args:
        model_name (str): Tên của model (ví dụ: 'add_youknow').
        device (str or torch.device): Thiết bị để tải model lên ('cpu' hoặc 'cuda').
        
    Returns:
        torch.nn.Module: Model đã được tải và sẵn sàng để sử dụng.
    """
    # --- SỬA LỖI QUAN TRỌNG: Xây dựng đường dẫn tương đối một cách chính xác ---
    # Lấy đường dẫn của file script này (model_loader.py)
    script_path = os.path.abspath(__file__)
    # Đi ngược lên 2 cấp để lấy thư mục gốc của dự án (từ script_AI/ -> UnOrCensored/)
    project_root = os.path.dirname(os.path.dirname(script_path))
    
    # Bây giờ, đường dẫn đến thư mục models sẽ luôn đúng
    models_dir = os.path.join(project_root, 'pre_trained_models')
    
    pth_path = os.path.join(models_dir, f"{model_name}.pth")
    yaml_path = os.path.join(models_dir, f"{model_name}_structure.yaml")

    print(f"Đang tìm file trọng số tại: {pth_path}")

    # --- Kiểm tra sự tồn tại của các file ---
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số model: {pth_path}")
    if not os.path.exists(yaml_path):
        print(f"Cảnh báo: Không tìm thấy file cấu trúc {yaml_path}. Vẫn tiếp tục tải model.")

    # --- Dựng lại kiến trúc model dựa vào tên ---
    print(f"Bắt đầu dựng kiến trúc cho model '{model_name}'...")
    model = None
    if model_name in ['add_youknow', 'mosaic_position']:
        model = models.BiSeNet(n_classes=1)
        print("Đã khởi tạo kiến trúc BiSeNet.")
    elif model_name == 'clean_youknow':
        model = models.InpaintingGenerator(in_channels=4, out_channels=3)
        print("Đã khởi tạo kiến trúc InpaintingGenerator.")
    else:
        raise ValueError(f"Tên model không hợp lệ: '{model_name}'.")

    # --- Nạp trọng số từ file .pth ---
    try:
        print(f"Đang nạp trọng số từ: {pth_path}")
        state_dict = torch.load(pth_path, map_location=torch.device(device))
        
        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print("Nạp trọng số thành công.")
        
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi nạp trọng số model: {e}")
        raise e
        
    model.to(device)
    model.eval()
    
    print(f"Model '{model_name}' đã sẵn sàng trên thiết bị '{device}'.")
    return model

if __name__ == '__main__':
    # Để chạy test, bạn cần có các file model trong thư mục pre_trained_models
    try:
        print("\n--- Thử tải model 'add_youknow' ---")
        model_add = load_model('add_youknow')
        print(model_add)
    except Exception as e:
        print(f"Không thể tải 'add_youknow': {e}")


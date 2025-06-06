# UnOrCensored/train.py
import argparse
import os
import sys

# Thêm các đường dẫn của module vào sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'script_AI'))
sys.path.insert(0, os.path.join(project_root, 'script_AI', 'train'))

def main(args):
   """
   Hàm điều phối chính cho việc huấn luyện hoặc tinh chỉnh (fine-tune) các model.
   """
   print(f"Bắt đầu quá trình huấn luyện cho model: '{args.pre_trained_models_name}'")
   
   # Xác định đường dẫn dữ liệu
   if args.folder_path:
       data_path = args.folder_path
       if not os.path.isdir(data_path):
           print(f"Lỗi: Thư mục dữ liệu '{data_path}' không tồn tại.")
           sys.exit(1)
   else:
       # Sử dụng đường dẫn mặc định
       default_data_dir = os.path.join(project_root, 'data')
       # Tên model map với tên thư mục dữ liệu (ví dụ: clean_youknow -> clean_youknow)
       model_data_folder = args.pre_trained_models_name
       data_path = os.path.join(default_data_dir, model_data_folder)
       print(f"Sử dụng thư mục dữ liệu mặc định: {data_path}")
       if not os.path.isdir(data_path):
           print(f"Lỗi: Thư mục dữ liệu mặc định '{data_path}' không tồn tại.")
           print("Vui lòng tạo thư mục và đặt dữ liệu vào theo đúng cấu trúc dự án,")
           print("hoặc sử dụng tham số --folder_path để chỉ định đường dẫn.")
           sys.exit(1)
           
   # Xác định tên script huấn luyện tương ứng
   # ví dụ: add_youknow -> train_add_youknow.py
   train_script_name = f"train_{args.pre_trained_models_name}"
   
   try:
       print(f"Đang tìm và gọi script huấn luyện: {train_script_name}")
       train_module = __import__(train_script_name)
   except ImportError as e:
       print(f"Lỗi: Không thể tìm thấy script huấn luyện '{train_script_name}.py'.")
       print(f"Chi tiết lỗi: {e}")
       print("Hãy chắc chắn rằng script tồn tại trong thư mục 'script_AI/train/'.")
       sys.exit(1)
       
   # Gọi hàm main của script huấn luyện với các tham số cần thiết
   # Giả sử các script train đều có hàm main nhận vào các đường dẫn cần thiết
   try:
       # Các script train sẽ tự xử lý việc load dữ liệu từ data_path
       # và load model từ pre_trained_models
       train_module.main([
           '--data_path', data_path,
           '--model_name', args.pre_trained_models_name
           # có thể thêm các tham số khác như epochs, batch_size...
       ])
       print(f"Quá trình huấn luyện cho model '{args.pre_trained_models_name}' đã hoàn tất.")
   except Exception as e:
       print(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")


if __name__ == '__main__':
   parser = argparse.ArgumentParser(
       description="Script chính để huấn luyện (fine-tune) các model AI.",
       formatter_class=argparse.RawTextHelpFormatter
   )
   parser.add_argument(
       '--pre_trained_models_name',
       type=str,
       required=True,
       choices=['add_youknow', 'mosaic_position', 'clean_youknow'],
       help="Tên của model cần huấn luyện/tinh chỉnh."
   )
   parser.add_argument(
       '--folder_path',
       type=str,
       required=False,
       help="(Tùy chọn) Đường dẫn đến thư mục chứa dữ liệu huấn luyện.\n"
            "Nếu không cung cấp, script sẽ sử dụng thư mục mặc định trong 'data/'."
   )

   if len(sys.argv) == 1:
       parser.print_help(sys.stderr)
       sys.exit(1)
       
   args = parser.parse_args()
   main(args)
# UnOrCensored/run.py
import argparse
import os
import sys
import shutil
import json
from datetime import datetime

# Thêm các đường dẫn của module vào sys.path để có thể import
# Điều này rất quan trọng để các script có thể tìm thấy nhau
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'tool'))
sys.path.insert(0, os.path.join(project_root, 'script_AI'))
sys.path.insert(0, os.path.join(project_root, 'script_AI', 'run'))


# Import các module cần thiết từ các thư mục con
# Sử dụng try-except để bắt lỗi import và đưa ra hướng dẫn rõ ràng
try:
   import get_file_type
   import get_file_info
   import split_video
   import video_to_frames
   import frames_to_video
   import merge_video
   # Các module AI sẽ được import động bên trong hàm main
except ImportError as e:
   print(f"Lỗi import: {e}")
   print("Hãy chắc chắn rằng bạn đã chạy script từ thư mục gốc 'UnOrCensored'")
   print("Và cấu trúc thư mục của dự án là chính xác.")
   sys.exit(1)


def main(args):
   """
   Hàm điều phối chính, thực hiện toàn bộ luồng xử lý ảnh/video.
   """
   start_time = datetime.now()
   print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu tác vụ '{args.task_name}' cho file: {args.file_path}")

   # --- Giai đoạn 1: Phân tích & Chuẩn bị Môi trường ---
   if not os.path.exists(args.file_path):
       print(f"Lỗi: File không tồn tại tại đường dẫn '{args.file_path}'")
       return

   # Tạo các thư mục output và tmp nếu chưa có
   output_dir = os.path.join(project_root, 'output')
   tmp_dir = os.path.join(project_root, 'tmp')
   os.makedirs(output_dir, exist_ok=True)
   if os.path.exists(tmp_dir):
       shutil.rmtree(tmp_dir) # Dọn dẹp tmp từ lần chạy trước
   os.makedirs(tmp_dir)

   file_type_result = get_file_type.main(['--file_path', args.file_path])
   file_type = file_type_result['file_type']
   print(f"Phát hiện loại file: {file_type}")
   
   # --- Xử lý theo từng loại file ---
   if file_type == 'image':
       process_image(args, tmp_dir, output_dir)
   elif file_type == 'video':
       process_video(args, tmp_dir, output_dir)
   else:
       print(f"Lỗi: Loại file '{file_type}' không được hỗ trợ.")
       return

   # --- Giai đoạn cuối: Dọn dẹp và di chuyển kết quả ---
   final_filename = f"{os.path.splitext(os.path.basename(args.file_path))[0]}_{args.task_name}_result{os.path.splitext(args.file_path)[1]}"
   final_output_path = os.path.join(output_dir, final_filename)
   
   if args.folder_path:
       if not os.path.isdir(args.folder_path):
           print(f"Lỗi: Đường dẫn thư mục '{args.folder_path}' không tồn tại.")
       else:
           try:
               shutil.move(final_output_path, os.path.join(args.folder_path, final_filename))
               print(f"Đã di chuyển file kết quả tới: {os.path.join(args.folder_path, final_filename)}")
           except Exception as e:
               print(f"Lỗi khi di chuyển file kết quả: {e}")

   # Dọn dẹp thư mục tmp
   print("Dọn dẹp các file tạm thời...")
   shutil.rmtree(tmp_dir)
   
   end_time = datetime.now()
   print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Tác vụ hoàn thành!")
   print(f"Tổng thời gian thực thi: {end_time - start_time}")


def process_image(args, tmp_dir, output_dir):
   """Xử lý cho file ảnh."""
   print("Bắt đầu xử lý ảnh...")
   # ... (Code xử lý ảnh sẽ được thêm vào đây)
   # Tạm thời sao chép ảnh gốc để minh họa luồng chạy
   # Trong thực tế sẽ gọi module AI tương ứng
   
   # Import module AI động
   ai_module_name = f"run_{args.task_name.replace('mosaic', 'youknow')}"
   try:
       ai_module = __import__(ai_module_name)
   except ImportError:
       print(f"Lỗi: Không tìm thấy module AI '{ai_module_name}.py'.")
       return

   final_filename = f"{os.path.splitext(os.path.basename(args.file_path))[0]}_{args.task_name}_result.png"
   output_image_path = os.path.join(output_dir, final_filename)
   
   # Gọi hàm main của module AI
   print(f"Gọi module AI: {ai_module_name}")
   ai_module.main(['--input_path', args.file_path, '--output_path', output_image_path])
   
   print(f"Ảnh đã xử lý được lưu tại: {output_image_path}")

def process_video(args, tmp_dir, output_dir):
   """Xử lý cho file video."""
   print("Bắt đầu xử lý video...")
   
   # 1. Trích xuất metadata
   print("Trích xuất metadata video...")
   info_result = get_file_info.main(['--file_path', args.file_path])
   with open(os.path.join(tmp_dir, 'video_metadata.json'), 'w') as f:
       json.dump(info_result, f, indent=4)
   print("Đã lưu metadata video.")

   # 2. Chia video thành các chunk (phần này sẽ phức tạp hơn theo tài liệu)
   print("Chia video thành các chunk...")
   split_video.main(['--file_path', args.file_path, '--output_dir', tmp_dir])
   
   video_chunks = sorted([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.startswith('chunk_')])
   processed_chunks = []

   # Import module AI động
   ai_module_name = f"run_{args.task_name.replace('mosaic', 'youknow')}"
   try:
       ai_module = __import__(ai_module_name)
   except ImportError:
       print(f"Lỗi: Không tìm thấy module AI '{ai_module_name}.py'.")
       return

   for i, chunk_path in enumerate(video_chunks):
       print(f"--- Xử lý chunk {i+1}/{len(video_chunks)}: {os.path.basename(chunk_path)} ---")
       chunk_process_dir = os.path.join(tmp_dir, f"chunk_{i}_processing")
       os.makedirs(chunk_process_dir, exist_ok=True)
       
       # 3. Trích xuất frames và audio
       video_to_frames.main(['--video_path', chunk_path, '--output_dir', chunk_process_dir])
       
       frames_dir = os.path.join(chunk_process_dir, 'frames')
       processed_frames_dir = os.path.join(chunk_process_dir, 'processed_frames')
       os.makedirs(processed_frames_dir, exist_ok=True)

       # 4. Xử lý từng frame bằng AI
       print(f"Gọi module AI '{ai_module_name}' cho các frame...")
       ai_module.main(['--input_path', frames_dir, '--output_path', processed_frames_dir])

       # 5. Ghép frames thành video chunk đã xử lý
       processed_chunk_path = os.path.join(tmp_dir, f"processed_chunk_{i}.mp4")
       frames_to_video.main([
           '--frames_dir', processed_frames_dir,
           '--audio_path', os.path.join(chunk_process_dir, 'audio.aac'),
           '--output_path', processed_chunk_path,
           '--metadata_path', os.path.join(tmp_dir, 'video_metadata.json')
       ])
       processed_chunks.append(processed_chunk_path)
       
       # Dọn dẹp thư mục xử lý của chunk
       shutil.rmtree(chunk_process_dir)
       print(f"--- Hoàn thành xử lý chunk {i+1} ---")

   # 6. Nối các video chunk đã xử lý
   print("Nối các video chunk đã xử lý...")
   final_filename = f"{os.path.splitext(os.path.basename(args.file_path))[0]}_{args.task_name}_result.mp4"
   final_output_path = os.path.join(output_dir, final_filename)
   merge_video.main(['--chunks_dir', tmp_dir, '--output_path', final_output_path])
   
   print(f"Video kết quả đã được lưu tại: {final_output_path}")


if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description="Script chính để xử lý ảnh/video với các tác vụ AI.",
       formatter_class=argparse.RawTextHelpFormatter
   )
   parser.add_argument(
       '--file_path',
       type=str,
       required=True,
       help="Đường dẫn đầy đủ đến file ảnh hoặc video cần xử lý."
   )
   parser.add_argument(
       '--task_name',
       type=str,
       required=True,
       choices=['add_mosaic', 'clean_mosaic'],
       help="Tên tác vụ cần thực hiện:\n"
            "  add_mosaic: Thêm mosaic vào đối tượng.\n"
            "  clean_mosaic: Làm sạch mosaic trên đối tượng."
   )
   parser.add_argument(
       '--folder_path',
       type=str,
       required=False,
       help="(Tùy chọn) Đường dẫn đến thư mục để lưu kết quả cuối cùng.\n"
            "Nếu không cung cấp, file sẽ được lưu trong thư mục 'output/' của dự án."
   )
   
   # In help nếu không có tham số nào được truyền
   if len(sys.argv) == 1:
       parser.print_help(sys.stderr)
       sys.exit(1)
       
   args = parser.parse_args()
   main(args)
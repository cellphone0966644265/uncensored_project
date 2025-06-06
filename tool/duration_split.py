# UnOrCensored/tool/duration_split.py
import argparse
import sys
import os
import shutil
import subprocess
import json
import psutil

def calculate_optimal_duration(file_path, tmp_dir, fps):
   """
   Tính toán thời lượng (giây) tối ưu cho mỗi video chunk dựa trên dung lượng đĩa trống.
   Đây là một bước quản lý tài nguyên thông minh như mô tả trong tài liệu.
   
   Args:
       file_path (str): Đường dẫn video gốc.
       tmp_dir (str): Thư mục tạm để lưu frame mẫu.
       fps (float): Số khung hình trên giây của video.
       
   Returns:
       dict: Kết quả tính toán.
   """
   try:
       # 1. Trích xuất Frame Mẫu
       sample_frame_path = os.path.join(tmp_dir, 'sample_frame.png')
       command = [
           'ffmpeg',
           '-i', file_path,
           '-ss', '00:00:01', # Lấy frame ở giây đầu tiên
           '-vframes', '1',
           '-q:v', '2', # Chất lượng tốt
           '-y',
           sample_frame_path
       ]
       subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       
       if not os.path.exists(sample_frame_path):
           return {"success": False, "message": "Không thể trích xuất frame mẫu."}

       # 2. Đo Kích thước Frame Chính xác
       exact_frame_size = os.path.getsize(sample_frame_path) # tính bằng bytes
       
       # 3. Ước tính Tổng Dung lượng mỗi Frame
       # Con số này tính đến dung lượng cho 3 file: gốc, mask, và đã xử lý (đều là PNG)
       total_space_per_frame = exact_frame_size * 3
       
       # 4. Tính Tốc độ Dữ liệu (bytes/giây)
       data_rate = total_space_per_frame * fps
       
       # 5. Xác định Dung lượng An toàn
       # Lấy dung lượng đĩa trống của phân vùng chứa thư mục tmp
       disk_usage = shutil.disk_usage(tmp_dir)
       available_disk_space = disk_usage.free
       usable_space = available_disk_space * 0.80 # Chừa lại 20% dung lượng trống
       
       # 6. Tính Thời lượng Tối đa (giây)
       if data_rate == 0:
           return {"success": False, "message": "Tốc độ dữ liệu bằng 0, không thể chia."}
           
       optimal_chunk_duration = usable_space / data_rate
       
       # Xóa frame mẫu
       os.remove(sample_frame_path)

       return {
           "success": True,
           "optimal_chunk_duration_seconds": optimal_chunk_duration,
           "estimated_data_rate_bytes_per_sec": data_rate,
           "usable_disk_space_bytes": usable_space
       }

   except Exception as e:
       return {
           "success": False,
           "message": f"Lỗi không xác định: {str(e)}"
       }

def main(argv=None):
   parser = argparse.ArgumentParser(description="Tính toán thời lượng tối ưu cho mỗi video chunk.")
   parser.add_argument('--file_path', required=True, help="Đường dẫn video gốc.")
   parser.add_argument('--tmp_dir', required=True, help="Thư mục tạm để làm việc.")
   parser.add_argument('--fps', required=True, type=float, help="FPS của video.")
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   result = calculate_optimal_duration(args.file_path, args.tmp_dir, args.fps)
   
   print(json.dumps(result, indent=4))
   return result

if __name__ == "__main__":
   if len(sys.argv) == 1:
       sys.argv.append('-h')
   main()
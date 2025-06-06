# UnOrCensored/tool/cut_video.py
import argparse
import sys
import os
import subprocess

def cut_video(file_path, start_time, end_time, output_path):
   """
   Cắt video từ thời gian bắt đầu đến kết thúc mà không cần mã hóa lại (stream copy).
   
   Args:
       file_path (str): Đường dẫn video gốc.
       start_time (str): Thời gian bắt đầu (format HH:MM:SS).
       end_time (str): Thời gian kết thúc (format HH:MM:SS).
       output_path (str): Đường dẫn file video sau khi cắt.
       
   Returns:
       dict: Kết quả thực thi.
   """
   try:
       # Lệnh ffmpeg để cắt video, sử dụng -c copy để giữ nguyên chất lượng
       command = [
           'ffmpeg',
           '-i', file_path,
           '-ss', start_time,
           '-to', end_time,
           '-c', 'copy',      # Sao chép stream, không mã hóa lại
           '-y',              # Ghi đè file output nếu đã tồn tại
           output_path
       ]
       
       print(f"Thực thi lệnh: {' '.join(command)}")
       # Chạy lệnh và ẩn output không cần thiết
       subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       
       if os.path.exists(output_path):
           return {
               "success": True,
               "output_path": output_path,
               "message": "Cắt video thành công."
           }
       else:
            return {
               "success": False,
               "message": "Cắt video thất bại, file output không được tạo."
           }

   except subprocess.CalledProcessError as e:
       return {
           "success": False,
           "message": "Lỗi FFmpeg khi cắt video.",
           "error_details": e.stderr.decode('utf-8')
       }
   except Exception as e:
       return {
           "success": False,
           "message": f"Lỗi không xác định: {str(e)}"
       }

def main(argv=None):
   parser = argparse.ArgumentParser(description="Cắt một đoạn video mà không mã hóa lại.")
   parser.add_argument('--file_path', required=True, help="Đường dẫn video gốc.")
   parser.add_argument('--start_time', required=True, help="Thời gian bắt đầu (ví dụ: 00:01:20).")
   parser.add_argument('--end_time', required=True, help="Thời gian kết thúc (ví dụ: 00:02:00).")
   parser.add_argument('--output_path', required=True, help="Đường dẫn file video output.")
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   result = cut_video(args.file_path, args.start_time, args.end_time, args.output_path)
   
   print(result.get('message'))
   if not result.get('success'):
       print(result.get('error_details', ''))
       
   return result

if __name__ == "__main__":
   if len(sys.argv) == 1:
       sys.argv.append('-h')
   main()
# UnOrCensored/tool/merge_video.py
import argparse
import sys
import os
import subprocess

def merge_videos_concat_demuxer(chunks_dir, output_path):
   """
   Sử dụng phương pháp concat demuxer của FFmpeg để nối các file video.
   Đây là phương pháp hiệu quả nhất vì nó tránh việc mã hóa lại.
   Điều kiện: các video chunk phải có cùng codec và thông số.
   
   Args:
       chunks_dir (str): Thư mục chứa các video chunk đã xử lý (ví dụ: processed_chunk_*.mp4).
       output_path (str): Đường dẫn file video cuối cùng.
       
   Returns:
       dict: Kết quả thực thi.
   """
   try:
       # Tìm tất cả các chunk đã xử lý
       chunks = sorted([f for f in os.listdir(chunks_dir) if f.startswith('processed_chunk_') and f.endswith('.mp4')])
       
       if not chunks:
           return {"success": False, "message": "Không tìm thấy video chunk nào để nối."}
           
       # Tạo file list cho FFmpeg
       list_file_path = os.path.join(chunks_dir, 'concat_list.txt')
       with open(list_file_path, 'w') as f:
           for chunk in chunks:
               # FFmpeg yêu cầu định dạng 'file /path/to/file'
               f.write(f"file '{os.path.join(chunks_dir, chunk)}'\n")

       # Lệnh FFmpeg sử dụng concat demuxer
       command = [
           'ffmpeg',
           '-f', 'concat',
           '-safe', '0', # Cần thiết khi đường dẫn không an toàn
           '-i', list_file_path,
           '-c', 'copy', # Rất quan trọng: không mã hóa lại
           '-y',
           output_path
       ]
       
       print(f"Thực thi lệnh nối video: {' '.join(command)}")
       subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       
       # Dọn dẹp file list
       os.remove(list_file_path)

       return {"success": True, "message": f"Nối video thành công: {output_path}"}

   except subprocess.CalledProcessError as e:
       return {
           "success": False,
           "message": "Lỗi FFmpeg khi nối video.",
           "error_details": e.stderr.decode('utf-8')
       }
   except Exception as e:
       return {"success": False, "message": f"Lỗi không xác định: {str(e)}"}

def main(argv=None):
   parser = argparse.ArgumentParser(description="Nối nhiều video chunk thành một file duy nhất mà không mã hóa lại.")
   parser.add_argument('--chunks_dir', required=True, help="Thư mục chứa các video chunk (ví dụ: processed_chunk_*.mp4).")
   parser.add_argument('--output_path', required=True, help="Đường dẫn file video kết quả.")
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   result = merge_videos_concat_demuxer(args.chunks_dir, args.output_path)
   print(result.get('message'))
   if not result.get('success'):
       print(result.get('error_details', ''))
       
   return result

if __name__ == "__main__":
   if len(sys.argv) == 1:
       sys.argv.append('-h')
   main()
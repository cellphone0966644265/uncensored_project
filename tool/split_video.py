# UnOrCensored/tool/split_video.py
import argparse
import sys
import os
import subprocess
import json

# Import các tool khác
try:
   import get_file_info
   import duration_split
except ImportError:
   print("Lỗi: Không thể import các module 'get_file_info' hoặc 'duration_split'.")
   print("Hãy chắc chắn bạn đang chạy từ thư mục gốc và các file tool tồn tại.")
   sys.exit(1)

def split_video_by_duration(file_path, output_dir, chunk_duration):
   """
   Sử dụng FFmpeg để chia video thành nhiều chunk nhỏ hơn.
   Sử dụng stream copy (-c copy) để không mã hóa lại.
   
   Args:
       file_path (str): Đường dẫn video gốc.
       output_dir (str): Thư mục để lưu các chunk.
       chunk_duration (int): Thời lượng của mỗi chunk (giây).
       
   Returns:
       dict: Kết quả thực thi.
   """
   try:
       if not os.path.exists(output_dir):
           os.makedirs(output_dir)
           
       output_template = os.path.join(output_dir, "chunk_%04d" + os.path.splitext(file_path)[1])
       
       command = [
           'ffmpeg',
           '-i', file_path,
           '-c', 'copy',
           '-map', '0',
           '-segment_time', str(chunk_duration),
           '-f', 'segment',
           '-reset_timestamps', '1',
           '-y',
           output_template
       ]
       
       print(f"Thực thi lệnh: {' '.join(command)}")
       subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       
       chunks = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('chunk_')])
       
       return {
           "success": True,
           "message": f"Chia video thành công thành {len(chunks)} chunks.",
           "chunks_list": chunks
       }

   except subprocess.CalledProcessError as e:
       return {
           "success": False,
           "message": "Lỗi FFmpeg khi chia video.",
           "error_details": e.stderr.decode('utf-8')
       }
   except Exception as e:
       return {
           "success": False,
           "message": f"Lỗi không xác định: {str(e)}"
       }

def main(argv=None):
   parser = argparse.ArgumentParser(
       description="Chia video thành các phần nhỏ hơn dựa trên thời lượng tính toán tự động."
   )
   parser.add_argument('--file_path', required=True, help="Đường dẫn video gốc.")
   parser.add_argument('--output_dir', required=True, help="Thư mục để lưu các chunk.")
   parser.add_argument(
       '--duration', type=int, required=False, 
       help="(Tùy chọn) Chỉ định thời lượng (giây) cho mỗi chunk. Nếu bỏ qua, sẽ tự động tính toán."
   )
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   if args.duration:
       chunk_duration = args.duration
       print(f"Sử dụng thời lượng chunk được chỉ định: {chunk_duration} giây.")
   else:
       print("Bắt đầu tự động tính toán thời lượng chunk tối ưu...")
       info = get_file_info.main(['--file_path', args.file_path])
       if 'error' in info or info.get('fps', 0) == 0:
           print(f"Lỗi khi lấy thông tin video: {info.get('error')}")
           return
           
       calc_result = duration_split.main([
           '--file_path', args.file_path,
           '--tmp_dir', args.output_dir, # Dùng luôn thư mục output làm thư mục tạm
           '--fps', str(info['fps'])
       ])
       
       if not calc_result.get("success"):
           print(f"Không thể tính toán thời lượng tối ưu: {calc_result.get('message')}")
           return
           
       chunk_duration = int(calc_result.get("optimal_chunk_duration_seconds", 60))
       print(f"Thời lượng chunk tối ưu được tính toán là: {chunk_duration} giây.")

   result = split_video_by_duration(args.file_path, args.output_dir, chunk_duration)
   print(json.dumps(result, indent=4))
   
   return result

if __name__ == "__main__":
   if len(sys.argv) == 1:
       sys.argv.append('-h')
   main()
# UnOrCensored/tool/frames_to_video.py
import argparse
import sys
import os
import subprocess
import json

def compile_frames_to_video(frames_dir, audio_path, output_path, metadata_path):
   """
   Biên dịch các frame PNG đã xử lý thành video, sử dụng lại metadata gốc
   và ghép lại file audio tương ứng.
   
   Args:
       frames_dir (str): Thư mục chứa các frame PNG đã xử lý.
       audio_path (str): Đường dẫn đến file audio đã tách ra (có thể None).
       output_path (str): Đường dẫn file video output.
       metadata_path (str): Đường dẫn đến file JSON chứa metadata của video gốc.
       
   Returns:
       dict: Kết quả thực thi.
   """
   try:
       with open(metadata_path, 'r') as f:
           metadata = json.load(f)
           
       video_stream = metadata.get('video_stream', {})
       fps = metadata.get('fps', 30)
       pixel_format = video_stream.get('pix_fmt', 'yuv420p')
       video_codec = video_stream.get('codec_name', 'libx264')
       
       # Bắt đầu xây dựng lệnh FFmpeg
       command = [
           'ffmpeg',
           '-framerate', str(fps),
           '-i', os.path.join(frames_dir, 'frame_%08d.png'),
       ]
       
       # Thêm audio nếu có
       if audio_path and os.path.exists(audio_path):
           command.extend(['-i', audio_path])
           # Chỉ định codec cho audio và video
           # -map 0:v -> lấy video từ input đầu tiên
           # -map 1:a -> lấy audio từ input thứ hai
           command.extend(['-c:v', video_codec, '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0'])
       else:
           # Chỉ có video
           command.extend(['-c:v', video_codec])
           
       # Thêm các tham số khác để giữ chất lượng
       command.extend([
           '-pix_fmt', pixel_format,
           '-shortest', # Kết thúc khi stream ngắn nhất (audio) kết thúc
           '-y',
           output_path
       ])
       
       print(f"Thực thi lệnh ghép video: {' '.join(command)}")
       subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

       return {"success": True, "message": f"Ghép video thành công: {output_path}"}

   except FileNotFoundError:
       return {"success": False, "message": f"Lỗi: Không tìm thấy file metadata tại {metadata_path}"}
   except subprocess.CalledProcessError as e:
       return {
           "success": False,
           "message": "Lỗi FFmpeg khi ghép video.",
           "error_details": e.stderr.decode('utf-8')
       }
   except Exception as e:
       return {"success": False, "message": f"Lỗi không xác định: {str(e)}"}

def main(argv=None):
   parser = argparse.ArgumentParser(description="Ghép các frame PNG và file audio thành một video hoàn chỉnh.")
   parser.add_argument('--frames_dir', required=True, help="Thư mục chứa các frame (ví dụ: frame_%08d.png).")
   parser.add_argument('--audio_path', required=False, help="(Tùy chọn) Đường dẫn đến file audio.")
   parser.add_argument('--output_path', required=True, help="Đường dẫn file video output.")
   parser.add_argument('--metadata_path', required=True, help="Đường dẫn đến file .json chứa metadata gốc.")
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   result = compile_frames_to_video(args.frames_dir, args.audio_path, args.output_path, args.metadata_path)
   print(result.get('message'))
   if not result.get('success'):
       print(result.get('error_details', ''))
       
   return result

if __name__ == "__main__":
   if len(sys.argv) == 1:
       sys.argv.append('-h')
   main()
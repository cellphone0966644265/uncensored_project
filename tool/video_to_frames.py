# UnOrCensored/tool/video_to_frames.py
import argparse
import sys
import os
import subprocess
import shutil

def extract_frames_and_audio(video_path, output_dir):
   """
   Giải mã video thành các frame ảnh PNG và tách file audio gốc ra.
   
   Args:
       video_path (str): Đường dẫn đến video chunk.
       output_dir (str): Thư mục để lưu frames và audio.
       
   Returns:
       dict: Kết quả thực thi.
   """
   try:
       frames_dir = os.path.join(output_dir, 'frames')
       audio_path = os.path.join(output_dir, 'audio.aac') # Sử dụng định dạng aac phổ biến
       os.makedirs(frames_dir, exist_ok=True)
       
       # Lệnh trích xuất frames thành file PNG
       # -q:v 2 là chất lượng tốt cho PNG
       frame_command = [
           'ffmpeg',
           '-i', video_path,
           '-q:v', '2',
           os.path.join(frames_dir, 'frame_%08d.png'),
           '-y'
       ]
       
       print(f"Trích xuất frames từ {os.path.basename(video_path)}...")
       subprocess.run(frame_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       
       # Lệnh trích xuất audio, sử dụng stream copy để giữ nguyên chất lượng
       # -vn: video disable
       # -acodec copy: audio codec copy
       audio_command = [
           'ffmpeg',
           '-i', video_path,
           '-vn',
           '-acodec', 'copy',
           audio_path,
           '-y'
       ]
       
       print(f"Trích xuất audio từ {os.path.basename(video_path)}...")
       # Sử dụng try-except riêng cho audio vì video có thể không có audio
       try:
           subprocess.run(audio_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
           audio_extracted = os.path.exists(audio_path)
       except subprocess.CalledProcessError:
           audio_extracted = False
           print("Cảnh báo: Không tìm thấy hoặc không thể trích xuất audio stream.")

       num_frames = len(os.listdir(frames_dir))
       
       return {
           "success": True,
           "message": f"Trích xuất thành công {num_frames} frames và audio.",
           "frames_dir": frames_dir,
           "audio_path": audio_path if audio_extracted else None,
       }

   except subprocess.CalledProcessError as e:
       return {
           "success": False,
           "message": "Lỗi FFmpeg khi trích xuất frames/audio.",
           "error_details": e.stderr.decode('utf-8')
       }
   except Exception as e:
       return {
           "success": False,
           "message": f"Lỗi không xác định: {str(e)}"
       }

def main(argv=None):
   parser = argparse.ArgumentParser(description="Trích xuất video thành các frame ảnh PNG và file audio.")
   parser.add_argument('--video_path', required=True, help="Đường dẫn đến file video.")
   parser.add_argument('--output_dir', required=True, help="Thư mục để lưu frames và audio.")
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   result = extract_frames_and_audio(args.video_path, args.output_dir)
   print(result.get('message'))
   if not result.get('success'):
       print(result.get('error_details', ''))
       
   return result

if __name__ == "__main__":
   if len(sys.argv) == 1:
       sys.argv.append('-h')
   main()
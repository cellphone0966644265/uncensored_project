# UnOrCensored/tool/get_file_info.py
import argparse
import sys
import json
import ffmpeg

def get_video_info(file_path):
   """
   Sử dụng ffprobe để trích xuất toàn bộ metadata của một file video.
   Đây là bước cực kỳ quan trọng để đảm bảo chất lượng video sau khi xử lý.

   Args:
       file_path (str): Đường dẫn đến file video.

   Returns:
       dict: Một dictionary chứa toàn bộ thông tin của video.
             Trả về dict chứa lỗi nếu có vấn đề xảy ra.
   """
   try:
       # Sử dụng ffprobe để lấy thông tin chi tiết
       probe = ffmpeg.probe(file_path)
       
       # Tìm video stream và audio stream
       video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
       audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)

       if video_stream is None:
           return {"error": f"Không tìm thấy video stream trong file: {file_path}"}
           
       info = {
           "file_path": file_path,
           "format": probe.get('format', {}),
           "video_stream": video_stream,
           "audio_stream": audio_stream # Có thể là None nếu không có audio
       }
       
       # Trích xuất một vài thông số quan trọng để dễ truy cập
       info['width'] = video_stream.get('width')
       info['height'] = video_stream.get('height')
       
       # Tính toán FPS (có thể ở dạng "30/1" hoặc "29.97")
       if 'avg_frame_rate' in video_stream and video_stream['avg_frame_rate'] != '0/0':
           num, den = map(int, video_stream['avg_frame_rate'].split('/'))
           info['fps'] = num / den if den != 0 else 0
       else:
           info['fps'] = 0 # Hoặc một giá trị mặc định

       info['duration'] = float(video_stream.get('duration', 0))
       
       return info

   except ffmpeg.Error as e:
       return {"error": f"Lỗi FFmpeg: {e.stderr.decode('utf8')}"}
   except Exception as e:
       return {"error": f"Lỗi không xác định: {str(e)}"}

def main(argv=None):
   parser = argparse.ArgumentParser(description="Trích xuất metadata chi tiết từ một file video/audio.")
   parser.add_argument(
       '--file_path',
       type=str,
       required=True,
       help="Đường dẫn đến file video cần lấy thông tin."
   )
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   result = get_video_info(args.file_path)
   
   # In ra stdout dưới dạng JSON
   print(json.dumps(result, indent=4))
   
   return result

if __name__ == "__main__":
   if len(sys.argv) == 1:
       sys.argv.append('-h')
   main()
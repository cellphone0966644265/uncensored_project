# UnOrCensored/tool/get_file_type.py
import mimetypes
import argparse
import sys
import json

def get_file_type(file_path):
   """
   Xác định loại file (image, video, audio, unknown) dựa vào mimetype.
   
   Args:
       file_path (str): Đường dẫn đến file.
       
   Returns:
       dict: Một dictionary chứa loại file và đường dẫn.
   """
   try:
       mimetypes.init()
       mimestart = mimetypes.guess_type(file_path)[0]

       if mimestart is not None:
           mimestart = mimestart.split('/')[0]
           if mimestart in ['image', 'video', 'audio']:
               file_type = mimestart
           else:
               file_type = 'unknown'
       else:
           file_type = 'unknown'
           
       return {
           "file_path": file_path,
           "file_type": file_type
       }

   except Exception as e:
       return {
           "file_path": file_path,
           "error": str(e)
       }

def main(argv=None):
   parser = argparse.ArgumentParser(description="Xác định loại file (ảnh, video, audio).")
   parser.add_argument(
       '--file_path',
       type=str,
       required=True,
       help="Đường dẫn đến file cần kiểm tra."
   )
   
   # Phân tích các tham số từ argv nếu được cung cấp, ngược lại từ sys.argv
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])
   
   result = get_file_type(args.file_path)

   # In ra stdout dưới dạng JSON để các script khác có thể gọi và sử dụng
   print(json.dumps(result, indent=4))
   
   # Trả về kết quả dạng dict để có thể gọi trực tiếp từ script khác
   return result

if __name__ == "__main__":
   if len(sys.argv) == 1:
       sys.argv.append('-h')
   main()
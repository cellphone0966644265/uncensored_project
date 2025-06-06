# UnOrCensored/run.py
import argparse
import os
import sys
import shutil
import json
from datetime import datetime

# (Các phần import khác giữ nguyên...)
# Thêm các đường dẫn của module vào sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'tool'))
sys.path.insert(0, os.path.join(project_root, 'script_AI'))
sys.path.insert(0, os.path.join(project_root, 'script_AI', 'run'))

try:
    import get_file_type
    import get_file_info
    import cut_video
    import split_video
    import video_to_frames
    import frames_to_video
    import merge_video
except ImportError as e:
    print(f"Lỗi import: {e}")
    sys.exit(1)


def main(args):
    start_time_proc = datetime.now()
    print(f"[{start_time_proc.strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu tác vụ '{args.task_name}' cho file: {args.file_path}")

    if not os.path.exists(args.file_path):
        print(f"Lỗi: File không tồn tại tại đường dẫn '{args.file_path}'")
        return

    output_dir = os.path.join(project_root, 'output')
    tmp_dir = os.path.join(project_dir, 'tmp')
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    file_type_result = get_file_type.main(['--file_path', args.file_path])
    file_type = file_type_result['file_type']
    print(f"Phát hiện loại file: {file_type}")

    file_to_process = args.file_path

    # --- Giai đoạn 1.5: Cắt video nếu có tham số thời gian (Logic đã được đơn giản hóa) ---
    if file_type == 'video' and args.start_time != "00:00:00":
        print("\n--- Nhận diện tham số thời gian, gọi module cắt video ---")
        temp_cut_video_path = os.path.join(tmp_dir, f"temp_cut_{os.path.basename(args.file_path)}")
        
        # Truyền thẳng tham số xuống cho cut_video.py, module này sẽ tự xử lý logic
        cut_result = cut_video.main([
            '--file_path', args.file_path,
            '--start_time', args.start_time,
            '--end_time', args.end_time, # Truyền cả end_time, dù nó là "00:00:00"
            '--output_path', temp_cut_video_path
        ])
        
        if cut_result.get("success"):
            file_to_process = temp_cut_video_path
            print(f"✅ Đã cắt video thành công. File tạm thời sẽ được xử lý: {file_to_process}\n")
        else:
            print("❌ Lỗi khi cắt video. Sẽ bỏ qua và xử lý toàn bộ video gốc.\n")
    
    # (Phần xử lý image/video và dọn dẹp giữ nguyên như trước...)
    # ...
    # ...
    # (Hàm process_image và process_video giữ nguyên)
    # ...
    # ...
    
    # Phần parser giữ nguyên như phiên bản trước
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    main(args)


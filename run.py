# UnOrCensored/run.py
import argparse
import os
import sys
import shutil
import json
from datetime import datetime

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
    print(f"❌ Lỗi import: {e}")
    sys.exit(1)


def process_image_pipeline(args, tmp_dir, output_dir):
    """
    Pipeline xử lý dành riêng cho file ảnh.
    Bỏ qua các bước không cần thiết của video.
    """
    print("\n--- Bắt đầu pipeline xử lý ảnh ---")
    
    # Bước 1: Chuẩn hóa input. Tạo thư mục tạm và sao chép ảnh vào đó.
    input_frame_dir = os.path.join(tmp_dir, "input_frames")
    os.makedirs(input_frame_dir)
    ext = os.path.splitext(args.file_path)[1]
    temp_image_path = os.path.join(input_frame_dir, f"frame_00000001{ext}")
    shutil.copy(args.file_path, temp_image_path)
    print(f"📄 Đã chuẩn hóa input, ảnh được đặt tại: {temp_image_path}")

    # Bước 2: Chạy module AI
    processed_frames_dir = os.path.join(tmp_dir, "processed_frames")
    os.makedirs(processed_frames_dir)
    ai_module_name = f"run_{args.task_name.replace('mosaic', 'youknow')}"
    try:
        ai_module = __import__(ai_module_name)
        print(f"🤖 Đang gọi module AI: {ai_module_name}")
        ai_module.main(['--input_path', input_frame_dir, '--output_path', processed_frames_dir])
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi chạy module AI: {e}")
        return

    # Bước 3: Di chuyển thẳng file kết quả đến output
    final_filename = f"{os.path.splitext(os.path.basename(args.file_path))[0]}_{args.task_name}{ext}"
    final_output_path = os.path.join(output_dir, final_filename)
    
    processed_files = os.listdir(processed_frames_dir)
    if processed_files:
        processed_image_path = os.path.join(processed_frames_dir, processed_files[0])
        shutil.copy(processed_image_path, final_output_path)
        print(f"✅ Đã lưu ảnh kết quả tại: {final_output_path}")
    else:
        print("❌ Lỗi: Không tìm thấy file kết quả sau khi xử lý AI.")

def process_video_pipeline(args, tmp_dir, output_dir):
    """
    Pipeline xử lý đầy đủ dành riêng cho file video.
    """
    print("\n--- Bắt đầu pipeline xử lý video ---")
    file_to_process = args.file_path

    # Bước 1: Cắt video nếu cần
    if args.start_time != "00:00:00":
        print("🕒 Nhận diện tham số thời gian, gọi module cắt video...")
        temp_cut_video_path = os.path.join(tmp_dir, f"temp_cut_{os.path.basename(args.file_path)}")
        cut_result = cut_video.main(['--file_path', file_to_process, '--start_time', args.start_time, '--end_time', args.end_time, '--output_path', temp_cut_video_path])
        if cut_result.get("success"):
            file_to_process = temp_cut_video_path
        else:
            print("⚠️ Lỗi khi cắt video. Sẽ xử lý toàn bộ video.")
    
    # Bước 2: Lấy metadata gốc
    metadata = get_file_info.main(['--file_path', args.file_path])
    metadata_path = os.path.join(tmp_dir, 'video_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
        
    # Bước 3: Chia video thành các chunks
    split_video.main(['--file_path', file_to_process, '--output_dir', tmp_dir])
    video_chunks = sorted([os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.startswith('chunk_')])
    
    if not video_chunks:
        print("❌ Lỗi: Không thể chia video thành các chunks.")
        return

    # Bước 4: Xử lý từng chunk
    processed_chunks = []
    ai_module_name = f"run_{args.task_name.replace('mosaic', 'youknow')}"
    try:
        ai_module = __import__(ai_module_name)
    except Exception as e:
        print(f"❌ Lỗi khi tải module AI '{ai_module_name}': {e}")
        return

    for i, chunk_path in enumerate(video_chunks):
        print(f"\n--- Xử lý chunk {i+1}/{len(video_chunks)} ---")
        chunk_process_dir = os.path.join(tmp_dir, f"chunk_{i}_processing")
        os.makedirs(chunk_process_dir)

        frames_result = video_to_frames.main(['--video_path', chunk_path, '--output_dir', chunk_process_dir])
        input_frames_dir = frames_result.get('frames_dir')
        audio_path = frames_result.get('audio_path')
        
        processed_frames_dir = os.path.join(chunk_process_dir, 'processed_frames')
        os.makedirs(processed_frames_dir)
        ai_module.main(['--input_path', input_frames_dir, '--output_path', processed_frames_dir])

        processed_chunk_path = os.path.join(tmp_dir, f"processed_chunk_{i}.mp4")
        frames_to_video.main(['--frames_dir', processed_frames_dir, '--audio_path', audio_path if audio_path and os.path.exists(audio_path) else "",'--output_path', processed_chunk_path,'--metadata_path', metadata_path])
        processed_chunks.append(processed_chunk_path)
        shutil.rmtree(chunk_process_dir)

    # Bước 5: Ghép các chunk đã xử lý
    final_filename = f"{os.path.splitext(os.path.basename(args.file_path))[0]}_{args.task_name}{os.path.splitext(args.file_path)[1]}"
    final_output_path = os.path.join(output_dir, final_filename)
    merge_video.main(['--chunks_dir', tmp_dir, '--output_path', final_output_path])
    print(f"✅ Đã lưu video kết quả tại: {final_output_path}")

def main(args):
    start_time_proc = datetime.now()
    print(f"[{start_time_proc.strftime('%Y-%m-%d %H:%M:%S')}] Bắt đầu tác vụ '{args.task_name}' cho file: {args.file_path}")

    # Giai đoạn 1: Kiểm tra và thiết lập môi trường
    if not os.path.exists(args.file_path):
        print(f"❌ Lỗi: File không tồn tại tại '{args.file_path}'")
        return

    output_dir = os.path.join(project_root, 'output')
    tmp_dir = os.path.join(project_root, 'tmp')
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    # Giai đoạn 2: Phân tích loại file và gọi pipeline tương ứng
    file_type = get_file_type.main(['--file_path', args.file_path])['file_type']
    print(f"\n🔍 Phân tích loại file: {file_type}")

    if file_type == 'image':
        process_image_pipeline(args, tmp_dir, output_dir)
    elif file_type == 'video':
        process_video_pipeline(args, tmp_dir, output_dir)
    else:
        print(f"❌ Lỗi: Loại file '{file_type}' không được hỗ trợ.")
    
    # Giai đoạn cuối: Di chuyển kết quả nếu có folder_path
    final_filename = f"{os.path.splitext(os.path.basename(args.file_path))[0]}_{args.task_name}{os.path.splitext(args.file_path)[1]}"
    final_output_path_in_proj = os.path.join(output_dir, final_filename)
    
    if args.folder_path and os.path.isdir(args.folder_path):
        if os.path.exists(final_output_path_in_proj):
            try:
                shutil.move(final_output_path_in_proj, os.path.join(args.folder_path, final_filename))
                print(f"🚚 Đã di chuyển file kết quả tới: {args.folder_path}")
            except Exception as e:
                print(f"⚠️ Lỗi khi di chuyển file kết quả: {e}")

    # Dọn dẹp thư mục tạm
    if os.path.exists(tmp_dir):
        print("🧹 Dọn dẹp các file tạm thời...")
        shutil.rmtree(tmp_dir)
    
    end_time_proc = datetime.now()
    print(f"\n[{end_time_proc.strftime('%Y-%m-%d %H:%M:%S')}] Tác vụ hoàn thành!")
    print(f"Tổng thời gian thực thi: {end_time_proc - start_time_proc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script chính để xử lý ảnh/video với các tác vụ AI.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--file_path', type=str, required=True, help="Đường dẫn đầy đủ đến file ảnh hoặc video cần xử lý.")
    parser.add_argument('--task_name', type=str, required=True, choices=['add_mosaic', 'clean_mosaic'], help="Tên tác vụ cần thực hiện:\n  add_mosaic: Thêm mosaic vào đối tượng.\n  clean_mosaic: Làm sạch mosaic trên đối tượng.")
    parser.add_argument('--folder_path', type=str, required=False, help="(Tùy chọn) Đường dẫn đến thư mục để lưu kết quả cuối cùng.\nNếu không cung cấp, file sẽ được lưu trong thư mục 'output/' của dự án.")
    parser.add_argument('--start_time', type=str, default="00:00:00", help="(Tùy chọn) Thời gian bắt đầu xử lý video (format HH:MM:SS). Mặc định là 00:00:00.")
    parser.add_argument('--end_time', type=str, default="00:00:00", help="(Tùy chọn) Thời gian kết thúc xử lý video (format HH:MM:SS). Mặc định là '00:00:00' (xử lý đến hết).")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    main(args)


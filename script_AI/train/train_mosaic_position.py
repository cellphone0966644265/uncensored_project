# UnOrCensored/script_AI/train/train_mosaic_position.py
# Script này có cấu trúc gần như y hệt train_add_youknow.py
# vì chúng cùng sử dụng kiến trúc BiSeNet và dataset dạng cặp (ảnh, mask).
# Điểm khác biệt duy nhất là nguồn dữ liệu đầu vào.

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import model_loader

# Sử dụng lại cùng một lớp Dataset
class SegmentationDataset(Dataset):
   def __init__(self, image_dir, mask_dir, transform=None):
       self.image_dir = image_dir
       self.mask_dir = mask_dir
       self.transform = transform
       self.images = sorted(os.listdir(image_dir))

   def __len__(self):
       return len(self.images)

   def __getitem__(self, idx):
       img_name = self.images[idx]
       img_path = os.path.join(self.image_dir, img_name)
       mask_path = os.path.join(self.mask_dir, img_name)

       image = Image.open(img_path).convert("RGB")
       mask = Image.open(mask_path).convert("L")
       mask = np.array(mask)
       mask = (mask > 128).astype(np.float32)
       mask = Image.fromarray(mask)

       if self.transform:
           image = self.transform(image)
       
       mask = transforms.ToTensor()(mask)
       return image, mask

def main(argv=None):
   parser = argparse.ArgumentParser(description="Huấn luyện model 'mosaic_position'.")
   parser.add_argument('--data_path', required=True, help="Đường dẫn đến thư mục dữ liệu (data/mosaic_position).")
   parser.add_argument('--model_name', required=True, help="Tên model để load ('mosaic_position').")
   parser.add_argument('--epochs', type=int, default=10, help="Số lượng epochs.")
   parser.add_argument('--batch_size', type=int, default=4, help="Batch size.")
   parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # --- Chuẩn bị Dữ liệu ---
   # Thay đổi nguồn dữ liệu đầu vào theo yêu cầu của tác vụ này
   image_dir = os.path.join(args.data_path, 'mosaiced_images')
   mask_dir = os.path.join(args.data_path, 'mosaic_masks')
   
   transform = transforms.Compose([
       transforms.Resize((512, 512)),
       transforms.ToTensor(),
       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
   ])
   
   dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
   dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

   # --- Tải Model, Optimizer, Loss ---
   model = model_loader.load_model(args.model_name, device=device)
   model.train()

   optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
   criterion = nn.BCEWithLogitsLoss()

   print(f"--- Bắt đầu huấn luyện model '{args.model_name}' trên thiết bị '{device}' ---")
   # Vòng lặp huấn luyện giống hệt train_add_youknow.py
   for epoch in range(args.epochs):
       epoch_loss = 0
       progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
       for images, masks in progress_bar:
           images, masks = images.to(device), masks.to(device)
           outputs = model(images)
           loss = criterion(outputs, masks)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           epoch_loss += loss.item()
           progress_bar.set_postfix({'Loss': loss.item()})
       print(f"Epoch {epoch+1} - Average Loss: {epoch_loss / len(dataloader):.4f}")

   # --- Lưu Model ---
   output_model_path = os.path.join("pre_trained_models", f"{args.model_name}_finetuned.pth")
   torch.save(model.state_dict(), output_model_path)
   print(f"Đã huấn luyện xong. Model mới được lưu tại: {output_model_path}")

if __name__ == '__main__':
   if len(sys.argv) == 1:
       sys.argv.extend(['-h'])
   main()
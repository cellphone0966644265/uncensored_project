# UnOrCensored/script_AI/train/train_clean_youknow.py
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

# --- Định nghĩa Dataset cho Inpainting ---
class InpaintingDataset(Dataset):
   """
   Dataset cho tác vụ inpainting (làm sạch mosaic).
   Đọc bộ ba: ảnh bị che mờ, mask vị trí, và ảnh gốc (ground truth).
   """
   def __init__(self, mosaiced_dir, mask_dir, original_dir, transform=None):
       self.mosaiced_dir = mosaiced_dir
       self.mask_dir = mask_dir
       self.original_dir = original_dir
       self.transform = transform
       self.images = sorted(os.listdir(mosaiced_dir))

   def __len__(self):
       return len(self.images)

   def __getitem__(self, idx):
       img_name = self.images[idx]
       mosaiced_path = os.path.join(self.mosaiced_dir, img_name)
       mask_path = os.path.join(self.mask_dir, img_name)
       original_path = os.path.join(self.original_dir, img_name)

       mosaiced_img = Image.open(mosaiced_path).convert("RGB")
       mask = Image.open(mask_path).convert("L")
       original_img = Image.open(original_path).convert("RGB")

       if self.transform:
           mosaiced_img = self.transform(mosaiced_img)
           original_img = self.transform(original_img)
       
       # Mask chỉ cần chuyển sang tensor
       mask_transform = transforms.Compose([
           transforms.Resize(mosaiced_img.shape[1:]), # Cùng kích thước với ảnh
           transforms.ToTensor()
       ])
       mask = mask_transform(mask)

       return mosaiced_img, mask, original_img

def main(argv=None):
   parser = argparse.ArgumentParser(description="Huấn luyện model 'clean_youknow' (inpainting).")
   parser.add_argument('--data_path', required=True, help="Đường dẫn đến thư mục dữ liệu (data/clean_youknow).")
   parser.add_argument('--model_name', required=True, help="Tên model để load ('clean_youknow').")
   parser.add_argument('--epochs', type=int, default=20, help="Số lượng epochs.")
   parser.add_argument('--batch_size', type=int, default=2, help="Batch size (nhỏ hơn do model lớn).")
   parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate.")
   
   args = parser.parse_args(argv if argv is not None else sys.argv[1:])

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # --- Chuẩn bị Dữ liệu ---
   mosaiced_dir = os.path.join(args.data_path, 'mosaiced_images')
   mask_dir = os.path.join(args.data_path, 'mosaic_masks')
   original_dir = os.path.join(args.data_path, 'original_images')
   
   # Transform cho ảnh: resize, to tensor, và normalize về [-1, 1]
   transform = transforms.Compose([
       transforms.Resize((256, 256)),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   
   dataset = InpaintingDataset(mosaiced_dir, mask_dir, original_dir, transform=transform)
   dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

   # --- Tải Model, Optimizer, Loss ---
   model = model_loader.load_model(args.model_name, device=device)
   model.train()

   optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
   # Sử dụng L1 Loss (Mean Absolute Error) thường cho kết quả tốt hơn (ít mờ) cho inpainting
   criterion = nn.L1Loss()

   print(f"--- Bắt đầu huấn luyện model '{args.model_name}' trên thiết bị '{device}' ---")
   for epoch in range(args.epochs):
       epoch_loss = 0
       progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
       
       for mosaiced, masks, originals in progress_bar:
           mosaiced = mosaiced.to(device)
           masks = masks.to(device)
           originals = originals.to(device)

           # Forward pass
           outputs = model(mosaiced, masks)
           
           # Chỉ tính loss trên vùng mask
           loss = criterion(outputs * masks, originals * masks)

           # Backward and optimize
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           epoch_loss += loss.item()
           progress_bar.set_postfix({'L1 Loss': loss.item()})

       print(f"Epoch {epoch+1} - Average L1 Loss: {epoch_loss / len(dataloader):.4f}")

   # --- Lưu Model ---
   output_model_path = os.path.join("pre_trained_models", f"{args.model_name}_finetuned.pth")
   torch.save(model.state_dict(), output_model_path)
   print(f"Đã huấn luyện xong. Model mới được lưu tại: {output_model_path}")

if __name__ == '__main__':
   if len(sys.argv) == 1:
       sys.argv.extend(['-h'])
   main()
# models/cau_hinh_C/buoc3_inference_cau_hinh_C.py
import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from peft import PeftModel

# Thêm thư mục hiện tại vào path để Python tìm thấy mask_generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mask_generator import generate_hand_bounding_box_mask

def nhom_tuoi(boneage_thang: float) -> str:
    if boneage_thang < 24:
        return "infant"
    elif boneage_thang < 72:
        return "young child"
    elif boneage_thang < 132:
        return "child"
    elif boneage_thang < 180:
        return "adolescent, pre-pubertal to early pubertal growth plates"
    else:
        return "adolescent, near-mature growth plates"

def tao_caption(gioi_tinh_bool: bool, boneage_thang: float) -> str:
    gioi_tinh = "male" if gioi_tinh_bool else "female"
    thang = float(boneage_thang)
    nam = thang / 12.0
    nhom = nhom_tuoi(thang)
    return (
        f"pediatric hand and wrist X-ray radiograph, {gioi_tinh} patient, "
        f"skeletal age approximately {thang:.0f} months ({nam:.1f} years), {nhom}, "
        f"grayscale radiographic image, anatomically accurate carpal bones and phalanges"
    )

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model", default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--controlnet_model", default="lllyasviel/sd-controlnet-canny")
    ap.add_argument("--lora_path", required=True, help="Đường dẫn thư mục lora weights đã train")
    ap.add_argument("--test_csv", required=True, help="Đường dẫn rsna_test.csv")
    ap.add_argument("--test_image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ids", nargs="+", required=True, help="Danh sách patient ID cần sinh ảnh")
    ap.add_argument("--n_variants", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--canny_low", type=int, default=100)
    ap.add_argument("--canny_high", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.test_csv)
    df["patient_ID"] = df["patient_ID"].astype(str)
    
    print("=== Nạp mô hình ControlNet Canny ===")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model, torch_dtype=torch.float16)
    
    print("=== Nạp Pipeline Stable Diffusion ControlNet Inpaint ===")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.pretrained_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    
    print("=== Nạp và tích hợp thích ứng LoRA ===")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)
    pipe.unet = pipe.unet.merge_and_unload()
    
    # Tối ưu hóa bộ nhớ chuyên sâu cho GPU 4GB VRAM
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    
    generator = torch.Generator(device="cuda")
    
    for pid in args.ids:
        row = df[df["patient_ID"] == pid]
        if row.empty:
            print(f"⚠️ Bỏ qua ID {pid} vì không tìm thấy thông tin.")
            continue
        row = row.iloc[0]
        caption = tao_caption(row["sex"] == "M", float(row["bone_age"]))
        
        img_path = os.path.join(args.test_image_dir, f"{pid}.png")
        if not os.path.exists(img_path):
            print(f"⚠️ Bỏ qua ID {pid} vì không tìm thấy tệp ảnh.")
            continue
            
        # 1. Đọc và resize ảnh gốc
        image = Image.open(img_path).convert("RGB").resize((args.resolution, args.resolution))
        
        # 2. Tự động sinh mask Bounding-box để khóa vùng bàn tay
        mask = generate_hand_bounding_box_mask(image, margin=15)
        
        # 3. Trích xuất Canny edge từ ảnh gốc để làm điều kiện dẫn đường hình học
        img_np = np.array(image)
        canny_np = cv2.Canny(img_np, args.canny_low, args.canny_high)
        # ControlNet Canny yêu cầu ảnh 3 kênh màu (RGB)
        canny_rgb = cv2.cvtColor(canny_np, cv2.COLOR_GRAY2RGB)
        control_image = Image.fromarray(canny_rgb)
        
        pid_out_dir = os.path.join(args.output_dir, pid)
        os.makedirs(pid_out_dir, exist_ok=True)
        
        # Lưu lại ảnh gốc và ảnh mask/control để đối chứng kiểm thử
        image.save(os.path.join(pid_out_dir, f"{pid}_original.png"))
        mask.save(os.path.join(pid_out_dir, f"{pid}_mask_bbox.png"))
        control_image.save(os.path.join(pid_out_dir, f"{pid}_control_canny.png"))
        
        print(f"\n=== Tiến hành Inpaint ID {pid} với ControlNet ===")
        for v in range(1, args.n_variants + 1):
            generator.manual_seed(args.seed + v)
            result = pipe(
                prompt=caption,
                image=image,
                mask_image=mask,
                control_image=control_image,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[0]
            
            out_path = os.path.join(pid_out_dir, f"{pid}_cleaned_{v}.png")
            result.save(out_path)
            print(f"  ✅ Đã lưu phiên bản {v}: {out_path}")
            
    print("\n=== HOÀN TẤT INFERENCE CẤU HÌNH C ===")

if __name__ == "__main__":
    main()

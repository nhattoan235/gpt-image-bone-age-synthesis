# ============================================================
# CẤU HÌNH B — GIAI ĐOẠN 3: INFERENCE (sinh ảnh so sánh với A)
# Dùng LoRA đã fine-tune (lora_epoch_2 - epoch cuối) + mask trắng
# toàn phần (GIỐNG HỆT Cấu hình A) — chỉ đổi model, giữ nguyên
# mọi yếu tố khác để so sánh khoa học "sạch".
#
# Caption dùng lúc inference: tái sử dụng đúng hàm tao_caption()
# từ buoc1_tao_caption.py, điền tuổi/giới tính THẬT của từng ID
# (lấy từ rsna_test_ground_truth.csv) — vì model được train bằng
# kiểu caption mô tả này, không phải câu lệnh "hãy xóa artifact".
#
# Cách chạy:
#   python buoc3_inference_cau_hinh_B.py \
#       --lora_path "...\lora_out_FULL\lora_epoch_2" \
#       --test_csv "D:\Hoctap\Doan_totnghiep\Dataset\rsna_test.csv" \
#       --test_image_dir "D:\...\boneage-test-dataset\boneage-test-dataset" \
#       --output_dir "D:\...\cau_hinh_B_ket_qua" \
#       --ids 4360 4362 4364 4504 4513 4543 --n_variants 3
# ============================================================

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel


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
    """Y HỆT hàm trong buoc1_tao_caption.py — giữ nhất quán train/inference."""
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
    ap.add_argument("--lora_path", required=True, help="Đường dẫn thư mục lora_epoch_2 (hoặc epoch khác)")
    ap.add_argument("--test_csv", required=True, help="rsna_test.csv (patient_ID, sex, bone_age)")
    ap.add_argument("--test_image_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ids", nargs="+", required=True, help="Danh sách patient ID, vd: 4360 4362 4364")
    ap.add_argument("--n_variants", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.test_csv)
    df["patient_ID"] = df["patient_ID"].astype(str)

    print("=== Đang tải pipeline gốc + LoRA đã fine-tune ===")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model, torch_dtype=torch.float16,
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)
    pipe.unet = pipe.unet.merge_and_unload()  # gộp LoRA vào UNet gốc để inference nhanh hơn

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"⚠️ Không bật được xformers ({e}) — vẫn tiếp tục.")

    pipe = pipe.to("cuda")

    generator = torch.Generator(device="cuda")

    for pid in args.ids:
        row = df[df["patient_ID"] == pid]
        if row.empty:
            print(f"⚠️ Không tìm thấy ground truth cho ID {pid}, bỏ qua.")
            continue
        row = row.iloc[0]
        gioi_tinh_bool = (row["sex"] == "M")
        boneage = float(row["bone_age"])
        caption = tao_caption(gioi_tinh_bool, boneage)

        img_path = os.path.join(args.test_image_dir, f"{pid}.png")
        if not os.path.exists(img_path):
            print(f"⚠️ Không tìm thấy ảnh {img_path}, bỏ qua.")
            continue

        image = Image.open(img_path).convert("RGB").resize((args.resolution, args.resolution))
        # Mask trắng toàn phần — GIỐNG HỆT Cấu hình A
        mask = Image.new("L", (args.resolution, args.resolution), 255)

        pid_out_dir = os.path.join(args.output_dir, pid)
        os.makedirs(pid_out_dir, exist_ok=True)
        image.save(os.path.join(pid_out_dir, f"{pid}_original_{args.resolution}.png"))

        print(f"\n=== ID {pid} — caption: {caption[:60]}... ===")
        for v in range(1, args.n_variants + 1):
            generator.manual_seed(args.seed + v)  # seed khác nhau mỗi bản, nhưng tái lập được
            result = pipe(
                prompt=caption,
                image=image,
                mask_image=mask,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).images[0]
            out_path = os.path.join(pid_out_dir, f"{pid}_cleaned_{v}.png")
            result.save(out_path)
            print(f"  ✅ Bản {v}/{args.n_variants} đã lưu: {out_path}")

    print("\n=== HOÀN TẤT INFERENCE CẤU HÌNH B ===")


if __name__ == "__main__":
    main()

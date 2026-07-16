# ============================================================
# CẤU HÌNH B — GIAI ĐOẠN 2, BƯỚC 2 (ĐÃ SỬA — mask trung tính)
# ============================================================
#
# [SỬA SO VỚI BẢN TRƯỚC]: Hàm sinh_mask_ngau_nhien() ở bản trước
# cố tình kéo tâm mask ra xa vùng trung tâm ảnh (né vùng bàn tay).
# Điều này KHÔNG đúng với thiết kế gốc trong Huong_cai_tien.docx —
# việc "né vùng giải phẫu" chỉ nên là đặc trưng RIÊNG của Cấu hình C
# (qua mask tường minh lúc inference), không phải hành vi được học
# sẵn vào model ngay từ lúc train ở Cấu hình B.
#
# Lý do quan trọng: Cấu hình C dùng CHUNG model đã fine-tune ở B
# (không train lại — xem Huong_cai_tien.docx: "Từ mô hình đã fine-tune
# ở B, thay mask trắng toàn phần bằng mask bounding-box..."). Nếu B
# đã được dạy né trung tâm ngay từ lúc train, thì cải thiện đo được
# ở C sau này sẽ bị NHIỄU BIẾN (confounded) — không thể khẳng định
# chắc chắn cải thiện đó đến từ mask tường minh lúc inference (đúng
# giả thuyết PRISM muốn kiểm chứng) hay từ hành vi đã "rò rỉ" sẵn từ
# training. Để giữ phép so sánh A/B/C sạch, mask lúc TRAIN ở B cần
# trung tính (phủ đều toàn ảnh, không né vị trí nào) — đúng cách các
# checkpoint SD Inpainting gốc (runwayml/stable-diffusion-inpainting)
# vẫn được huấn luyện.
# ============================================================

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model


# ------------------------------------------------------------
# CHIẾN LƯỢC MASK NGẪU NHIÊN — TRUNG TÍNH (không thiên vị vị trí)
# ------------------------------------------------------------
def sinh_mask_ngau_nhien(size: int) -> Image.Image:
    """Sinh 1 mask nhị phân (255 = vùng được sửa) cho ảnh vuông `size`.

    [ĐÃ SỬA] Chiến lược trung tính — mô phỏng cách train chuẩn của
    Stable Diffusion Inpainting gốc, KHÔNG né bất kỳ vùng nào (kể cả
    trung tâm ảnh, nơi thường là bàn tay). Mục đích duy nhất của B là
    domain-adaptation: dạy model hiểu ngữ nghĩa ảnh X-quang bàn tay
    (texture xương, mật độ mô, độ tương phản grayscale y tế) — CHƯA
    có ý định dạy model "biết vùng nào nên/không nên sửa". Việc kiểm
    soát vùng chỉnh sửa là trách nhiệm RIÊNG của Cấu hình C (qua mask
    tường minh lúc inference), giữ tách bạch 2 biến số thực nghiệm.

    Trộn 2 kiểu mask phổ biến trong training inpainting tổng quát:
    - 1-3 hình chữ nhật ngẫu nhiên, vị trí đều khắp ảnh (kể cả tâm)
    - Mask dạng nét vẽ tự do (free-form strokes) mô phỏng artifact
      không đều hình dạng
    Tỉ lệ vùng bị mask trong khoảng 10-40% diện tích ảnh — theo đúng
    thông lệ training LaMa/SD-Inpainting phổ biến.
    """
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)

    kieu = random.random()

    if kieu < 0.6:
        # ----- Hình chữ nhật ngẫu nhiên, vị trí ĐỀU KHẮP ảnh -----
        so_hcn = random.randint(1, 3)
        for _ in range(so_hcn):
            w = random.uniform(0.10, 0.35) * size
            h = random.uniform(0.10, 0.35) * size
            # Tâm lấy mẫu ĐỀU trên toàn bộ ảnh, không loại trừ vùng nào
            tam_x = random.uniform(0, size)
            tam_y = random.uniform(0, size)

            x0 = max(0, tam_x - w / 2)
            y0 = max(0, tam_y - h / 2)
            x1 = min(size - 1, tam_x + w / 2)
            y1 = min(size - 1, tam_y + h / 2)
            draw.rectangle([x0, y0, x1, y1], fill=255)

    else:
        # ----- Free-form strokes (nét vẽ ngẫu nhiên, mô phỏng artifact bất định hình) -----
        so_net = random.randint(1, 4)
        for _ in range(so_net):
            x, y = random.uniform(0, size), random.uniform(0, size)
            do_day = random.uniform(0.02, 0.08) * size
            so_diem = random.randint(4, 8)
            points = [(x, y)]
            for _ in range(so_diem):
                x += random.uniform(-0.2, 0.2) * size
                y += random.uniform(-0.2, 0.2) * size
                x = min(max(x, 0), size)
                y = min(max(y, 0), size)
                points.append((x, y))
            draw.line(points, fill=255, width=int(do_day), joint="curve")
            for px, py in points:
                r = do_day / 2
                draw.ellipse([px - r, py - r, px + r, py + r], fill=255)

    return mask


class BoneAgeInpaintDataset(Dataset):
    """Đọc metadata.jsonl (do buoc1_tao_caption.py sinh ra) + sinh mask
    ngẫu nhiên trung tính on-the-fly cho mỗi mẫu mỗi epoch."""

    def __init__(self, data_dir: str, tokenizer: CLIPTokenizer, resolution: int = 512,
                 max_train_samples=None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution

        meta_path = os.path.join(data_dir, "metadata.jsonl")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]

        if max_train_samples is not None:
            self.records = self.records[:max_train_samples]

        self.image_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.data_dir, rec["file_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        mask_pil = sinh_mask_ngau_nhien(self.resolution)
        mask = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0).unsqueeze(0)  # 1,H,W

        # masked_image: vùng mask -> 0 (giống input inpainting pipeline chuẩn)
        masked_image = image * (mask < 0.5)

        input_ids = self.tokenizer(
            rec["text"], truncation=True, padding="max_length",
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": image,
            "mask": mask,
            "masked_image": masked_image,
            "input_ids": input_ids,
        }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model", default="runwayml/stable-diffusion-inpainting")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=3e-5)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--checkpointing_steps", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20,
                     help="In loss mỗi N global step. Hạ xuống 1-5 khi chạy test subset nhỏ "
                          "để thấy được loss ngay, vì subset nhỏ có ít global step.")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )

    # ----- Load các thành phần từ pipeline gốc -----
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    # Đóng băng VAE + text encoder — chỉ fine-tune UNet qua LoRA (tiết kiệm VRAM)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # cross/self-attention của UNet
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"⚠️ Không bật được xformers ({e}) — vẫn tiếp tục nhưng sẽ tốn VRAM hơn.", flush=True)

    unet.enable_gradient_checkpointing()

    # [ĐÃ SỬA] Bỏ hẳn optimizer 8-bit — với chỉ 1.59M tham số LoRA, VRAM
    # tiết kiệm được là không đáng kể (~vài MB), trong khi rủi ro bất ổn
    # định số học của quantization 8-bit cao hơn hẳn. Đây là nguyên nhân
    # nghi ngờ chính gây phân kỳ (diverge) quan sát được ở checkpoint
    # 500-1000 của lần train trước — chuyển hẳn về AdamW 32-bit chuẩn.
    optimizer_cls = torch.optim.AdamW
    print("ℹ️ Dùng AdamW 32-bit chuẩn (không dùng bitsandbytes 8-bit) để tránh rủi ro bất ổn định số học.", flush=True)

    optimizer = optimizer_cls(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    dataset = BoneAgeInpaintDataset(
        args.data_dir, tokenizer, resolution=args.resolution,
        max_train_samples=args.max_train_samples,
    )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(accelerator.device, dtype=torch.float16)
    text_encoder.to(accelerator.device, dtype=torch.float16)

    # ----- RESUME: nạp lại checkpoint gần nhất nếu có -----
    global_step = 0
    start_epoch = 0
    ckpt_dirs = sorted(
        [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]),
    ) if os.path.isdir(args.output_dir) else []

    if ckpt_dirs:
        last_ckpt = os.path.join(args.output_dir, ckpt_dirs[-1])
        print(f"🔄 Tìm thấy checkpoint, resume từ: {last_ckpt}", flush=True)
        accelerator.load_state(last_ckpt)
        global_step = int(ckpt_dirs[-1].split("-")[-1])
        steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
        start_epoch = global_step // max(steps_per_epoch, 1)

    # ----- Training loop -----
    training_start_time = time.time()
    loss_history = []
    window_losses = []  # [MỚI] tích lũy loss TỪNG micro-batch trong 1 window log, để tính TB đại diện hơn
    best_smoothed_loss = float("inf")
    so_lan_vuot_nguong_lien_tiep = 0
    da_bi_phan_ky = False

    for epoch in range(start_epoch, args.num_train_epochs):
        if da_bi_phan_ky:
            break
        unet.train()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
                masked_image = batch["masked_image"].to(accelerator.device, dtype=torch.float16)
                mask = batch["mask"].to(accelerator.device, dtype=torch.float16)
                input_ids = batch["input_ids"].to(accelerator.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    masked_latents = vae.encode(masked_image).latent_dist.sample() * vae.config.scaling_factor
                    mask_resized = F.interpolate(mask, size=latents.shape[-2:])

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Input UNet inpainting: 9 kênh = noisy_latents(4) + mask(1) + masked_latents(4)
                unet_input = torch.cat([noisy_latents, mask_resized, masked_latents], dim=1)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                window_losses.append(loss.item())  # [MỚI] lưu lại loss từng micro-batch, không chỉ cái cuối

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.log_every == 0:
                    elapsed = time.time() - training_start_time
                    sec_per_step = elapsed / global_step

                    # [SỬA] Dùng loss TRUNG BÌNH cả window (nhiều micro-batch) thay vì
                    # chỉ 1 sample cuối cùng — giảm nhiễu do batch_size=1 + timestep
                    # ngẫu nhiên khiến loss dao động mạnh tự nhiên giữa các sample.
                    current_loss = sum(window_losses) / len(window_losses)
                    window_losses = []  # reset cho window tiếp theo
                    loss_history.append(current_loss)

                    smoothed = sum(loss_history[-5:]) / len(loss_history[-5:])
                    best_smoothed_loss = min(best_smoothed_loss, smoothed)

                    # [SỬA] Chỉ báo phân kỳ THẬT nếu vượt ngưỡng LIÊN TIẾP nhiều lần
                    # (không phải chỉ 1 điểm dữ liệu đơn lẻ — tránh báo động giả do
                    # dao động tự nhiên của diffusion loss theo timestep ngẫu nhiên)
                    vuot_nguong = len(loss_history) >= 10 and smoothed > 8 * best_smoothed_loss
                    so_lan_vuot_nguong_lien_tiep = so_lan_vuot_nguong_lien_tiep + 1 if vuot_nguong else 0
                    canh_bao = f"  ⚠️ vượt ngưỡng ({so_lan_vuot_nguong_lien_tiep}/3 lần liên tiếp)" if vuot_nguong else ""

                    print(f"[epoch {epoch}] step {global_step} loss_TB_window={current_loss:.4f} "
                          f"(TB mượt={smoothed:.4f}) | {sec_per_step:.2f} giây/step "
                          f"| đã chạy {elapsed/60:.1f} phút{canh_bao}", flush=True)

                    if so_lan_vuot_nguong_lien_tiep >= 3:
                        print("🚨 Phát hiện PHÂN KỲ THẬT (loss trung bình window vượt ngưỡng "
                              "3 LẦN LIÊN TIẾP, không phải dao động ngẫu nhiên đơn lẻ) "
                              "— DỪNG TRAINING SỚM. Lưu checkpoint hiện tại để kiểm tra/debug.", flush=True)
                        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}-PHANKY")
                        accelerator.save_state(ckpt_path)
                        da_bi_phan_ky = True
                        break

                if global_step % args.checkpointing_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(ckpt_path)
                    print(f"💾 Đã lưu checkpoint: {ckpt_path}", flush=True)

        # Lưu LoRA weights sau mỗi epoch (để dùng thử inference ngay cả khi chưa train xong)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(unet)
            unwrapped.save_pretrained(os.path.join(args.output_dir, f"lora_epoch_{epoch}"))
            print(f"✅ Đã lưu LoRA weights sau epoch {epoch}", flush=True)

    total_elapsed = time.time() - training_start_time
    sec_per_step_final = total_elapsed / max(global_step, 1)
    steps_per_epoch_full = math.ceil(12611 / (args.train_batch_size * args.gradient_accumulation_steps))
    est_full_seconds = steps_per_epoch_full * args.num_train_epochs * sec_per_step_final

    print("=== HOÀN TẤT TRAINING CẤU HÌNH B ===", flush=True)
    print(f"\n--- THỐNG KÊ TỐC ĐỘ (dùng để ước tính chạy full) ---", flush=True)
    print(f"Tổng thời gian chạy: {total_elapsed/60:.1f} phút cho {global_step} step", flush=True)
    print(f"Tốc độ: {sec_per_step_final:.2f} giây/step", flush=True)
    print(f"Ước tính chạy FULL 12.611 ảnh, {args.num_train_epochs} epoch: "
          f"{est_full_seconds/3600:.1f} giờ ({steps_per_epoch_full * args.num_train_epochs} step)", flush=True)


if __name__ == "__main__":
    main()
# ============================================================
# CẤU HÌNH A — PHIÊN BẢN 2: CÓ SET quality="high" (theo mô tả Methods trong bài báo)
# Bám sát 100% logic gốc từ generate_mult.py (tác giả Matsuoka)
# Chỉ đổi: đường dẫn (Windows -> Colab), giới hạn còn 2 ID, API key nhập tay
# ============================================================

import os
import time
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
from getpass import getpass

# ----- [KHÁC BIỆT so với gốc]: nhập key an toàn thay vì hardcode -----
api_key = getpass("Nhập OpenAI API key: ")
client = OpenAI(api_key=api_key)

# ----- [KHÁC BIỆT so với gốc]: đường dẫn Colab thay vì Windows local -----
input_dir = f"{original_dir}/boneage-test-dataset/boneage-test-dataset"
output_dir = os.path.join(drive_path, "cau_hinh_A_quality_high")
SIZE = (1024, 1024)
N_SYNTHETIC = 3  # === GIỐNG HỆT GỐC: Number of synthetic images to generate per original

os.makedirs(output_dir, exist_ok=True)

# === GIỐNG HỆT GỐC — nguyên văn prompt, không đổi 1 chữ ===
prompt = (
    "Enhance this pediatric hand X-ray by digitally removing non-anatomical artifacts or labels outside the hand region, "
    "while preserving all anatomical details and ensuring the radiograph remains realistic and diagnostically useful."
)

# === GIỐNG HỆT GỐC — Prepare mask once ===
mask = Image.new("RGBA", SIZE, (255, 255, 255, 255))
mask_bytes = BytesIO()
mask.save(mask_bytes, format="PNG")
mask_bytes.seek(0)
mask_bytes.name = "mask.png"

# ----- [KHÁC BIỆT so với gốc]: giới hạn chỉ 2 ID thay vì toàn bộ 200 ảnh -----
selected_ids = ['4360', '4362']
image_files = [f"{pid}.png" for pid in selected_ids]

for idx, filename in enumerate(image_files):
    # === GIỐNG HỆT GỐC — Subdirectory for each original image ===
    base_name = os.path.splitext(filename)[0]
    img_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(img_output_dir, exist_ok=True)

    # === GIỐNG HỆT GỐC — Check if already processed (all 3 exist) ===
    already_done = all(
        os.path.exists(os.path.join(img_output_dir, f"{base_name}_cleaned_{i+1}.png"))
        for i in range(N_SYNTHETIC)
    )
    if already_done:
        print(f"Đã xử lý: {base_name}. Bỏ qua...")
        continue

    try:
        print(f"\nĐang xử lý ảnh {idx+1}/{len(image_files)}: {filename}")

        # === GIỐNG HỆT GỐC — resize KHÔNG chỉ định resample (mặc định PIL = BICUBIC) ===
        img = Image.open(os.path.join(input_dir, filename)).convert("RGB").resize(SIZE)
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        img_bytes.name = "image.png"

        mask_bytes.seek(0)

        # === 1 lần gọi API duy nhất với n=3, CÓ tham số quality="high" (khác code gốc, khớp bài báo) ===
        result = client.images.edit(
            model="gpt-image-1",
            image=img_bytes,
            mask=mask_bytes,
            prompt=prompt,
            n=N_SYNTHETIC,
            size="1024x1024"
            quality="high"  # <-- [KHÁC BIỆT DUY NHẤT so với code gốc]: thêm dòng này theo đúng mô tả Methods của bài báo
        )

        # === GIỐNG HỆT GỐC — Save each generated image ===
        for i, data in enumerate(result.data):
            image_base64 = data.b64_json
            image_bytes = base64.b64decode(image_base64)
            output_filename = f"{base_name}_cleaned_{i+1}.png"
            output_path = os.path.join(img_output_dir, output_filename)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            print(f"  Đã lưu: {output_path}")

    except Exception as e:
        print(f"Lỗi khi xử lý {filename}: {e}")

    # === GIỐNG HỆT GỐC — Adjust sleep time to avoid API rate limits ===
    time.sleep(15)

print("\n=== HOÀN TẤT — Phiên bản CÓ set quality=high ===")

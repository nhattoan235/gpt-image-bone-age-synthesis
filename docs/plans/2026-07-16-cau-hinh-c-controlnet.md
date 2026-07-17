# Cấu hình C (ControlNet Canny + Bounding-box Mask) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Xây dựng luồng tạo sinh ảnh X-quang bàn tay nâng cao (Cấu hình C) sử dụng mô hình lai Stable Diffusion Inpainting + LoRA + ControlNet Canny + Bounding-box Mask tự động để loại bỏ dị vật nhiễu mà không làm biến dạng xương tay. Đồng thời vá lỗi cú pháp hiện có ở Cấu hình A.

**Architecture:** 
1. Tự động hóa tạo mặt nạ (Bounding-box Mask): Phân đoạn vùng bàn tay bằng các kỹ thuật xử lý ảnh OpenCV (nhị phân hóa Otsu, tìm contour lớn nhất) để xác định hộp giới hạn (Bounding Box) của bàn tay, sau đó khóa vùng này (mask = 0) và chỉ cho phép inpaint ở các vùng nhiễu bên ngoài (mask = 255).
2. Tích hợp dẫn đường hình học: Dùng mô hình `ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")` kết hợp `StableDiffusionControlNetInpaintPipeline` của HuggingFace để cố định cấu trúc xương qua đường biên Canny của ảnh gốc.
3. Tối ưu bộ nhớ: Sử dụng các chế độ `enable_sequential_cpu_offload()` và `fp16` để chạy ổn định trên GPU RTX 3050 Ti 4GB VRAM.

**Tech Stack:** Python, PyTorch, Diffusers, Peft, OpenCV, PIL

---

### Task 1: Vá lỗi cú pháp thiếu dấu phẩy ở Cấu hình A

**Files:**
- Modify: [cau_hinh_A_v2_quality_high.py](file:///d:/Learning/DoAn%20Tot%20nghiep/gpt-image-bone-age-synthesis/models/cau_hinh_A/cau_hinh_A_v2_quality_high.py:78-81)
- Modify: [cau_hinh_A_chinh_thuc.py](file:///d:/Learning/DoAn%20Tot%20nghiep/gpt-image-bone-age-synthesis/models/cau_hinh_A/cau_hinh_A_chinh_thuc.py:78-81)

**Step 1: Vá lỗi thiếu dấu phẩy ở file v2**
Sửa phần gọi API `client.images.edit` trong file `cau_hinh_A_v2_quality_high.py`:
```python
        result = client.images.edit(
            model="gpt-image-1",
            image=img_bytes,
            mask=mask_bytes,
            prompt=prompt,
            n=N_SYNTHETIC,
            size="1024x1024",
            quality="high"
        )
```

**Step 2: Vá lỗi thiếu dấu phẩy ở file chính thức**
Sửa tương tự trong file `cau_hinh_A_chinh_thuc.py`.

**Step 3: Chạy kiểm tra cú pháp**
Kiểm tra tính hợp lệ của cú pháp Python bằng lệnh:
`python -m py_compile models/cau_hinh_A/cau_hinh_A_v2_quality_high.py models/cau_hinh_A/cau_hinh_A_chinh_thuc.py`
Expected: Không có lỗi biên dịch nào.

**Step 4: Commit thay đổi**
```bash
git add models/cau_hinh_A/cau_hinh_A_v2_quality_high.py models/cau_hinh_A/cau_hinh_A_chinh_thuc.py
git commit -m "fix: sửa lỗi cú pháp thiếu dấu phẩy khi gọi API gpt-image-1"
```

---

### Task 2: Phát triển mô-đun tạo Bounding-box Mask tự động bằng OpenCV

**Files:**
- Create: `models/cau_hinh_C/mask_generator.py`
- Test: `models/cau_hinh_C/test_mask_generator.py`

**Step 1: Viết test kiểm tra tính đúng đắn của tạo mask**
Tạo file test xác thực hàm sinh mask nhị phân trả về kích thước chính xác và vùng bàn tay được bảo vệ (giá trị 0).
```python
# models/cau_hinh_C/test_mask_generator.py
import numpy as np
from PIL import Image
from mask_generator import generate_hand_bounding_box_mask

def test_generate_hand_bounding_box_mask():
    # Tạo ảnh giả lập đen có một hình vuông trắng ở giữa (giả lập bàn tay)
    dummy_img = Image.new("L", (512, 512), 0)
    for x in range(150, 350):
        for y in range(150, 350):
            dummy_img.putpixel((x, y), 200)
    
    # Chuyển sang RGB
    dummy_img = dummy_img.convert("RGB")
    
    mask = generate_hand_bounding_box_mask(dummy_img, margin=10)
    mask_np = np.array(mask)
    
    assert mask.size == (512, 512)
    # Vùng trung tâm phải bằng 0 (bị che để bảo vệ)
    assert mask_np[250, 250] == 0
    # Rìa ảnh phải bằng 255 (cho phép inpaint xóa nhãn)
    assert mask_np[10, 10] == 255
    print("Test generate mask passed!")

if __name__ == "__main__":
    test_generate_hand_bounding_box_mask()
```

**Step 2: Viết mã nguồn tạo mask thông minh**
Tạo file `models/cau_hinh_C/mask_generator.py` sử dụng thuật toán phân vùng ảnh của OpenCV:
```python
# models/cau_hinh_C/mask_generator.py
import numpy as np
import cv2
from PIL import Image

def generate_hand_bounding_box_mask(image_pil: Image.Image, margin: int = 15) -> Image.Image:
    """
    Tự động phân đoạn bàn tay bằng OpenCV để tạo mask bounding box:
    - Vùng trong hộp giới hạn bàn tay: mask = 0 (bảo vệ cấu trúc giải phẫu)
    - Vùng ngoài hộp giới hạn (chứa nhãn dán, nhiễu): mask = 255 (cho phép inpaint làm sạch)
    """
    img_np = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Nhị phân hóa Otsu để tìm vùng bàn tay sáng màu trên nền tối
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tìm các contour liên kết
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Mặc định mask trắng toàn bộ (255)
    width, height = image_pil.size
    mask_np = np.full((height, width), 255, dtype=np.uint8)
    
    if contours:
        # Contour lớn nhất là vùng bàn tay
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Thêm biên an toàn (margin) để tránh mất sụn hoặc rìa da tay
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(width, x + w + margin)
        y1 = min(height, y + h + margin)
        
        # Ghi đè vùng bàn tay bằng màu đen (0) để bảo vệ giải phẫu
        mask_np[y0:y1, x0:x1] = 0
        
    return Image.fromarray(mask_np)
```

**Step 3: Chạy test kiểm tra**
Chạy script: `python models/cau_hinh_C/test_mask_generator.py`
Expected: In ra màn hình "Test generate mask passed!" và không báo lỗi.

**Step 4: Commit**
```bash
git add models/cau_hinh_C/mask_generator.py models/cau_hinh_C/test_mask_generator.py
git commit -m "feat: thêm mô-đun sinh mask bounding-box tự động bảo vệ bàn tay"
```

---

### Task 3: Phát triển mô-đun Inference Cấu hình C (ControlNet Canny + Bounding Box Mask)

**Files:**
- Create: `models/cau_hinh_C/buoc3_inference_cau_hinh_C.py`

**Step 1: Viết script Inference chính thức**
Tạo file `models/cau_hinh_C/buoc3_inference_cau_hinh_C.py` nạp SD Inpainting + ControlNet Canny + LoRA:
```python
# models/cau_hinh_C/buoc3_inference_cau_hinh_C.py
import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from peft import PeftModel
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
```

**Step 2: Đảm bảo cú pháp sạch lỗi**
Chạy: `python -m py_compile models/cau_hinh_C/buoc3_inference_cau_hinh_C.py`
Expected: Hoàn thành biên dịch không báo lỗi cú pháp.

**Step 3: Commit**
```bash
git add models/cau_hinh_C/buoc3_inference_cau_hinh_C.py
git commit -m "feat: hoàn thiện script chạy sinh ảnh Cấu hình C với ControlNet Canny"
```

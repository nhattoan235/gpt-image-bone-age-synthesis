# models/cau_hinh_C/test_mask_generator.py
import numpy as np
import sys
import os
# Thêm thư mục hiện tại vào path để Python tìm thấy mask_generator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

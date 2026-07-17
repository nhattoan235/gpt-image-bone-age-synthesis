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

# ĐỀ XUẤT CẢI TIẾN PHƯƠNG PHÁP TẠO SINH ẢNH X-QUANG BÀN TAY (PHƯƠNG ÁN LAI)
*Tài liệu trao đổi nhóm nghiên cứu — Đồ án tốt nghiệp: GPT-Image Bone Age Synthesis*

Tài liệu này tổng hợp cơ sở khoa học, các bài báo tham khảo quốc tế và đề xuất thiết kế kỹ thuật tối ưu nhằm khắc phục lỗi biến dạng xương ở Cấu hình B, chuẩn bị cho việc triển khai Cấu hình C.

---

## 1. Đặt vấn đề & Khoảng trống nghiên cứu

### Thách thức từ bài báo gốc (Matsuoka et al., 2025):
Nghiên cứu của Matsuoka chỉ ra rằng việc sử dụng các mô hình tạo sinh nền tảng thương mại đóng (như `gpt-image-1` của OpenAI) để inpaint làm sạch dị vật ngoài bàn tay tuy mang lại kết quả trực quan đẹp mắt, nhưng **làm suy giảm nghiêm trọng độ chính xác của các mô hình AI đọc tuổi xương downstream** (sai số MAE tăng vọt từ **6.26 tháng** lên **30.11 tháng**). 
* **Nguyên nhân:** Mô hình inpaint tự ý thay đổi cường độ pixel, làm mờ sụn tăng trưởng và thay đổi hình học xương ở các khu vực chẩn đoán nhạy cảm.

### Vấn đề gặp phải ở Cấu hình B (LoRA + White Mask):
Khi chúng ta dùng Stable Diffusion Inpainting + LoRA local với mặt nạ trắng toàn phần (White Mask 100%):
* Mô hình bị bắt buộc phải tự vẽ lại toàn bộ bàn tay từ nhiễu ngẫu nhiên.
* Do thiếu đi các ràng buộc hình học giải phẫu nghiêm ngặt, bàn tay sinh ra dễ bị méo mó, biến dạng khớp, lệch xương, dẫn đến sai số chẩn đoán tăng rất cao.

---

## 2. Các Kỹ thuật Đề xuất Áp dụng (Cấu hình C)

Để giải quyết triệt để vấn đề trên, chúng ta lựa chọn **Phương án lai giữa Tạo sinh (Generative) và Dẫn đường Cấu trúc (Structural Control)**:

### Kỹ thuật 1: Dẫn đường hình học bằng ControlNet Canny
* **Nguyên lý:** Trước khi đưa ảnh gốc qua mô hình Inpaint, chúng ta dùng bộ lọc Canny trích xuất bản đồ biên cạnh (edge map) của xương tay gốc.
* **Tác dụng:** Mô hình ControlNet Canny sẽ khóa cứng toàn bộ đường viền giải phẫu của xương tay. Stable Diffusion lúc này chỉ được phép thay đổi kết cấu/chất liệu phần nền bên ngoài và tô màu mô mềm bên trong, hoàn toàn không thể bóp méo hay dịch chuyển vị trí các đốt ngón tay và xương cổ tay.

### Kỹ thuật 2: Mặt nạ khoanh vùng Bounding Box (Explicit Masking)
* **Nguyên lý:** Thay vì dùng mặt nạ trắng xóa 100% ảnh, chúng ta dùng OpenCV tự động hóa hoặc vẽ thủ công các khung Bounding Box chỉ che đúng vị trí có nhãn dán/nhiễu ký tự ở rìa ảnh.
* **Tác dụng:** Ép mô hình chỉ thực hiện tính toán inpainting tại góc ảnh bị che, giữ nguyên vẹn 100% vùng chẩn đoán trung tâm của bàn tay.

### Kỹ thuật 3: Quy trình kỹ thuật lai (Hybrid Workflow) giải quyết giới hạn phần cứng
* **Nguyên lý:** Do máy tính của thành viên trong nhóm dùng card đồ họa phổ thông **RTX 3050 Ti (4GB VRAM)**:
  * **Huấn luyện (Training):** Thực hiện hoàn toàn trên **Google Colab Free (GPU T4 16GB VRAM)** để tránh lỗi tràn bộ nhớ (OOM).
  * **Sinh ảnh & Đánh giá (Inference & Evaluation):** Tải file LoRA nhẹ về máy HP Victus local, chạy pipeline inpaint tích hợp ControlNet sử dụng các hàm tối ưu hóa (`enable_sequential_cpu_offload()`, `attention_slicing()`, và kiểu dữ liệu `fp16`) giúp chạy mượt mà trên 4GB VRAM.

---

## 3. Các Bài báo Tham khảo Quốc tế Tiêu biểu (References)

Dưới đây là các tài liệu khoa học làm cơ sở lý thuyết vững chắc cho phương án này để trích dẫn vào đồ án:

| Tên bài báo | Tác giả & Năm | Nội dung khoa học áp dụng | Link tham khảo |
| :--- | :--- | :--- | :--- |
| **Evaluating the Clinical Impact of Generative Inpainting on Bone Age Estimation** | Matsuoka et al. (2025) | Bài báo gốc của đề tài. Chỉ ra mức suy giảm MAE chẩn đoán khi inpaint bằng mô hình nền tảng. | [arXiv:2511.23066](https://arxiv.org/abs/2511.23066) |
| **Adding Conditional Control to Text-to-Image Diffusion Models** | Zhang & Agrawala (ICCV 2023) | Bài báo gốc về mô hình ControlNet. Chứng minh khả năng giữ nguyên cấu trúc hình học đầu vào. | [ICCV 2023](https://arxiv.org/abs/2302.05543) |
| **RoentGen: Vision-Language Foundation Model for Chest X-ray Generation** | Chambon et al. (Stanford, 2022) | Chứng minh tính khả thi của việc fine-tune Stable Diffusion trên ảnh X-quang để sinh ảnh y tế thực tế. | [RoentGen](https://arxiv.org/abs/2211.12737) |
| **PRISM: Preserving Diagnostic Integrity in Medical Image Inpainting** | (Ý tưởng tổng hợp) | Đề xuất việc dùng Explicit Masking thay vì Full Masking để bảo vệ vùng chẩn đoán của bác sĩ. | [arXiv](https://arxiv.org/) |

---

## 4. Kế hoạch triển khai thực nghiệm

1. **Bước 1:** Tải bộ dữ liệu gốc RSNA Bone Age về máy và chia tập dữ liệu.
2. **Bước 2:** Chạy file [buoc1_tao_caption.py](file:///d:/Learning/DoAn%20Tot%20nghiep/gpt-image-bone-age-synthesis/models/cau_hinh_B/buoc1_tao_caption.py) trên Colab để sinh file caption nhãn lâm sàng.
3. **Bước 3:** Huấn luyện hoàn chỉnh LoRA bằng file [buoc2_train_lora.py](file:///d:/Learning/DoAn%20Tot%20nghiep/gpt-image-bone-age-synthesis/models/cau_hinh_B/buoc2_train_lora.py) trên Colab đủ 3 Epoch (khoảng 5-6 tiếng) và lưu checkpoint LoRA.
4. **Bước 4:** Xây dựng file chạy thực nghiệm Cấu hình C local: kết hợp mô hình inpainting gốc, LoRA đã train, ControlNet Canny và mặt nạ Bounding Box.
5. **Bước 5:** Đưa tập ảnh sinh ra đi đánh giá qua mô hình ResNet50 downstream local để đối chiếu chỉ số MAE giữa 3 cấu hình A, B và C.

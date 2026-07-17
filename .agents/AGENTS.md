# Quy tắc huấn luyện mô hình chuẩn Google (Google-Standard ML Engineering Rules)
*Dự án: GPT-Image Bone Age Synthesis*

Bộ quy tắc này được thiết lập để định hướng cho AI Agent và lập trình viên tuân thủ các tiêu chuẩn kỹ thuật máy học nghiêm ngặt của Google, tối ưu hóa cho môi trường phần cứng giới hạn (GPU RTX 3050 Ti 4GB VRAM local và GPU T4 Colab Free).

---

## 1. Tính tái lập là ưu tiên số một (Reproducibility First)
Để đảm bảo các kết quả nghiên cứu khoa học trong đồ án có thể tái lập và đối chứng chính xác:
* **Khóa Seed ngẫu nhiên:** Mọi đoạn code huấn luyện và inference phải đặt seed ngẫu nhiên đồng bộ cho toàn bộ thư viện:
  ```python
  import random
  import numpy as np
  import torch
  
  def set_seed(seed=42):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      # Nếu cần tính nhất quan tuyệt đối (chấp nhận đánh đổi tốc độ)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```
* **Lưu cấu hình thực nghiệm:** Mỗi lượt chạy (training run) phải lưu lại file cấu hình định dạng JSON/YAML chứa đầy đủ các siêu tham số (Hyperparameters) và phiên bản commit của Git để đối chiếu.

---

## 2. Quản lý bộ nhớ tối đa tránh lỗi OOM (Memory Management)
Phần cứng local bị giới hạn ở 4GB VRAM, do đó khi viết mã nguồn cần tuân thủ các quy tắc tiết kiệm bộ nhớ:
* **Khi huấn luyện (Colab GPU T4):**
  * Bắt buộc sử dụng mixed precision (`fp16` hoặc `bf16`).
  * Bật chế độ Gradient Checkpointing để tiết kiệm VRAM khi lan truyền ngược: `unet.enable_gradient_checkpointing()`.
  * Sử dụng Gradient Accumulation (tích lũy gradient) để mô phỏng batch size lớn thay vì tăng trực tiếp batch size vật lý.
* **Khi Inference (Local HP Victus 3050 Ti):**
  * Luôn chuyển mô hình sang kiểu dữ liệu `torch.float16`.
  * Kích hoạt cơ chế giải phóng bộ nhớ của HuggingFace Diffusers:
    ```python
    # Tải từng phần mô hình lên GPU khi cần thiết và đẩy lại sang RAM/CPU khi xong
    pipe.enable_sequential_cpu_offload() 
    # Chia nhỏ phép tính attention để không chiếm dụng lượng lớn bộ nhớ liên tục
    pipe.enable_attention_slicing()
    ```
  * Giải phóng bộ nhớ đệm CUDA sau mỗi ảnh hoặc mỗi batch nhỏ: `torch.cuda.empty_cache()`.

---

## 3. Theo dõi Loss và dừng sớm tránh phân kỳ (Early Stopping & Divergence Control)
Tuân thủ Quy tắc số 22 trong hướng dẫn kỹ nghệ ML của Google (Rule #22: Keep track of metrics):
* **Tính toán mượt (Smoothed Loss):** Do Batch Size nhỏ, Loss của từng step đơn lẻ dao động rất mạnh. Bắt buộc tính toán Loss trung bình theo một cửa sổ trượt (Rolling Window, ví dụ: 50 steps) để làm mượt dữ liệu trước khi đánh giá xu hướng.
* **Tự động dừng khi phân kỳ:** Nếu giá trị Loss trung bình mượt vượt quá 8 lần mức Loss tối thiểu tốt nhất trong lịch sử huấn luyện, liên tục trong 3 lần kiểm tra, hệ thống phải tự động ngắt huấn luyện (`early stopping`) và lưu checkpoint lỗi để phân tích ngoại lệ.

---

## 4. Kiểm tra dữ liệu tiền huấn luyện (Pre-flight Data Integrity Checks)
Trước khi khởi chạy huấn luyện (tránh việc chương trình chạy được vài tiếng rồi crash do lỗi dữ liệu):
* **Kiểm tra ánh xạ dữ liệu:** Viết mã nguồn kiểm tra đồng bộ giữa tệp CSV nhãn giải phẫu và tệp ảnh vật lý trong ổ cứng. Nếu phát hiện một ID thiếu ảnh tương ứng, chương trình phải ghi log cảnh báo chi tiết và bỏ qua hàng đó một cách an toàn thay vì crash chương trình.
* **Xác thực định dạng ảnh:** Xác minh độ phân giải, kênh màu (RGB/Grayscale) và các giá trị NaN/Inf trong ma trận điểm ảnh trước khi đưa vào Dataloader.

---

## 5. Tách biệt Module thiết kế (Modular Execution Architecture)
Duy trì sự độc lập giữa các thành phần mã nguồn:
* **Module 1 (Huấn luyện - Cloud):** Tập trung vào việc fine-tune LoRA, đầu ra duy nhất là file trọng số nhẹ (`lora_weights`).
* **Module 2 (Tạo ảnh - Local):** Tải mô hình inpainting gốc + LoRA đã train + mô hình ControlNet Canny local để tạo sinh tập ảnh thử nghiệm.
* **Module 3 (Đánh giá - Local):** Độc lập hoàn toàn với quá trình sinh ảnh. Mô hình ResNet50 downstream chỉ nhận thư mục ảnh đầu ra để tính toán chỉ số MAE/RMSE.

---

## 6. Quy trình Tự động gọi và Áp dụng các Skill bổ trợ (Custom Agent Skills)
AI Agent khi làm việc với dự án này bắt buộc phải tự động kích hoạt các kỹ năng (Skills) tương ứng trong các tình huống sau:

* **Khi chuẩn bị triển khai các bước code hoặc cấu trúc phức tạp:**
  * Kích hoạt skill `writing-plans` để lập kế hoạch triển khai chi tiết trước khi chỉnh sửa mã nguồn.
  * Kích hoạt skill `executing-plans` để thực thi kế hoạch theo từng phase/task có kiểm soát và xác thực.
* **Khi tối ưu hóa quy trình huấn luyện, log hệ thống hoặc lưu checkpoint:**
  * Kích hoạt các hướng dẫn từ skill `mlops-engineer` và `ml-engineer` để định hình thiết kế lưu trữ metadata cấu hình, kết nối TensorBoard/WandB ổn định và ghi log chuẩn y tế.
* **Khi xây dựng các mô-đun đánh giá mô hình downstream:**
  * Kích hoạt skill `evaluation` kết hợp `scikit-learn` để tính toán chính xác các chỉ số thống kê MAE, RMSE, độ lệch chuẩn (SD), độ ổn định liên phiên sinh (Inter-generation consistency), và vẽ biểu đồ phân phối cường độ pixel đối chứng.
* **Khi chương trình gặp lỗi biên dịch, lỗi logic hoặc crash (đặc biệt là lỗi tràn VRAM OOM):**
  * Kích hoạt skill `systematic-debugging` (hoặc `debugger`) để điều tra nguyên nhân gốc rễ một cách khoa học từ log hệ thống trước khi đề xuất bất kỳ dòng code sửa lỗi nào.


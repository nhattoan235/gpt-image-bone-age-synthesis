# ============================================================
# CẤU HÌNH B — GIAI ĐOẠN 2, BƯỚC 1: SINH CAPTION TỰ ĐỘNG
# Input : CSV chuẩn Kaggle RSNA train (cột: id, boneage, male)
# Output: metadata.jsonl (chuẩn HuggingFace datasets "imagefolder"),
#         đặt cùng thư mục với ảnh train để load trực tiếp bằng
#         `datasets.load_dataset("imagefolder", data_dir=...)`
#
# Ý tưởng caption: mô tả tuổi (quy đổi năm + nhóm phát triển) và
# giới tính, giống tinh thần report-conditioning của ChexGen —
# giúp mô hình học ngữ nghĩa lâm sàng cơ bản (tuổi xương liên
# quan trực tiếp đến hình thái sụn tăng trưởng) thay vì chỉ học
# "ảnh X-quang bàn tay" chung chung.
# ============================================================

import pandas as pd
import json
import os
import argparse


def nhom_tuoi(boneage_thang: float) -> str:
    """Quy đổi tháng tuổi -> mô tả nhóm phát triển xương, dùng ngôn ngữ
    tự nhiên thay vì chỉ số thô, để gần với cách caption y tế thật hơn."""
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


def tao_caption(row) -> str:
    gioi_tinh = "male" if bool(row["male"]) else "female"
    thang = float(row["boneage"])
    nam = thang / 12.0
    nhom = nhom_tuoi(thang)

    return (
        f"pediatric hand and wrist X-ray radiograph, {gioi_tinh} patient, "
        f"skeletal age approximately {thang:.0f} months ({nam:.1f} years), {nhom}, "
        f"grayscale radiographic image, anatomically accurate carpal bones and phalanges"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Đường dẫn CSV train RSNA (id, boneage, male)")
    ap.add_argument("--image_dir", required=True, help="Thư mục chứa ảnh train (để kiểm tra file tồn tại)")
    ap.add_argument("--image_ext", default=".png", help="Đuôi file ảnh, mặc định .png")
    ap.add_argument("--out", default=None, help="Đường dẫn output metadata.jsonl (mặc định: image_dir/metadata.jsonl)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Chuẩn hoá tên cột phòng trường hợp CSV có id dạng "id" hoặc chỉ số khác
    cols = {c.lower(): c for c in df.columns}
    assert "id" in cols and "boneage" in cols and "male" in cols, (
        f"CSV thiếu cột bắt buộc (id, boneage, male). Cột hiện có: {list(df.columns)}"
    )
    df = df.rename(columns={cols["id"]: "id", cols["boneage"]: "boneage", cols["male"]: "male"})

    out_path = args.out or os.path.join(args.image_dir, "metadata.jsonl")

    n_ok, n_missing = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            file_name = f"{int(row['id'])}{args.image_ext}"
            full_path = os.path.join(args.image_dir, file_name)
            if not os.path.exists(full_path):
                n_missing += 1
                continue

            record = {
                "file_name": file_name,
                "text": tao_caption(row),
                "boneage_months": float(row["boneage"]),
                "male": bool(row["male"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"✅ Đã ghi {n_ok} caption vào: {out_path}")
    if n_missing:
        print(f"⚠️ Bỏ qua {n_missing} dòng CSV vì không tìm thấy file ảnh tương ứng trong {args.image_dir}")


if __name__ == "__main__":
    main()

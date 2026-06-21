import os

# ==========================
# Thiết lập thư mục cache trên ổ F
# ==========================
os.environ["HF_HOME"] = r"F:\DoAn\datasets\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"F:\DoAn\datasets\hf_cache\datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"F:\DoAn\datasets\hf_cache\hub"

from datasets import load_dataset

# ==========================
# Thư mục lưu dataset
# ==========================
SAVE_DIR = r"F:\DoAn\datasets\VSASV_full"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("Bắt đầu tải VSASV Dataset...")
print("=" * 60)

# ==========================
# Tải toàn bộ dataset
# ==========================
dataset = load_dataset(
    "hustep-lab/VSASV-Dataset"
)

print("\nĐã tải xong từ Hugging Face.")
print(dataset)

print("\nĐang lưu dataset vào ổ F...")

# ==========================
# Lưu dataset
# ==========================
dataset.save_to_disk(SAVE_DIR)

print("=" * 60)
print("Hoàn thành!")
print(f"Dataset đã được lưu tại: {SAVE_DIR}")
print("=" * 60)
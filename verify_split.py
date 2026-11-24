
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SPLIT_DIR = BASE_DIR / "dataset_split"

splits = ["train", "validation", "test"]
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

if not SPLIT_DIR.exists():
    print("ERROR: dataset_split folder not found!")
    raise SystemExit

total_images = 0

for split in splits:
    split_path = SPLIT_DIR / split

    print(f"\n--- {split.upper()} ---")

    if not split_path.exists():
        print(f"ERROR: {split} folder is missing!")
        continue

    categories = [folder for folder in split_path.iterdir() if folder.is_dir()]
    split_count = 0

    for category in sorted(categories):
        images = [
            file for file in category.iterdir()
            if file.is_file() and file.suffix.lower() in image_extensions
        ]

        print(f"{category.name}: {len(images)} images")
        split_count += len(images)

    print(f"Total {split} images: {split_count}")
    print(f"Number of categories: {len(categories)}")

    total_images += split_count

print("\n==============================")
print("DATASET SPLIT VERIFICATION")
print("==============================")
print("Total images:", total_images)

if total_images == 20638:
    print("SUCCESS: All original images are accounted for!")
else:
    print("WARNING: Image count differs from the original dataset.")

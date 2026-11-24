
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# 1. Project folder location
BASE_DIR = Path(__file__).resolve().parent

# 2. Original dataset location
SOURCE_DIR = BASE_DIR / "dataset" / "PlantVillage"

# 3. New dataset location
OUTPUT_DIR = BASE_DIR / "dataset_split"

# 4. Image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 5. Check original dataset
if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {SOURCE_DIR}"
    )

# 6. Collect images from all categories
image_paths = []
labels = []

for folder in sorted(SOURCE_DIR.iterdir()):

    if folder.is_dir():

        images = [
            file for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image in images:
            image_paths.append(image)
            labels.append(folder.name)

print("Total images:", len(image_paths))
print("Total categories:", len(set(labels)))

# 7. Check dataset availability
if len(image_paths) == 0:
    raise ValueError("No images found in the dataset!")

# 8. Check if split already exists
if OUTPUT_DIR.exists():

    response = input(
        "\ndataset_split already exists. "
        "Do you want to recreate it? (yes/no): "
    ).strip().lower()

    if response != "yes":
        print("Existing dataset split was not changed.")
        raise SystemExit

    shutil.rmtree(OUTPUT_DIR)

# 9. Split into training and remaining images
train_paths, remaining_paths, train_labels, remaining_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )
)

# 10. Split remaining images into validation and testing
val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        remaining_paths,
        remaining_labels,
        test_size=0.50,
        random_state=42,
        stratify=remaining_labels
    )
)

# 11. Organize the splits
splits = {
    "train": (train_paths, train_labels),
    "validation": (val_paths, val_labels),
    "test": (test_paths, test_labels)
}

# 12. Copy images into folders
for split_name, (paths, category_labels) in splits.items():

    for image, label in zip(paths, category_labels):

        destination_folder = OUTPUT_DIR / split_name / label

        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            image,
            destination_folder / image.name
        )

    print(f"{split_name}: {len(paths)} images")

# 13. Final result
print("\n================================")
print("DATASET SPLITTING COMPLETED!")
print("================================")

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Testing images:", len(test_paths))

print("Total images:", len(image_paths))
print("Output location:", OUTPUT_DIR)
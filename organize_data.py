# organize_data.py

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
SOURCE_DIR = Path(r'D:\AIML\Autism_Detection_Project\dataset\all')  # Path to the 'all' directory
DEST_DIR = Path(r'D:\AIML\Autism_Detection_Project\dataset')       # Path to the 'dataset' directory

CLASSES = ['autism', 'neurotypical']
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1
RANDOM_STATE = 42  # For reproducibility

def create_directories(dest_path):
    """Create train, validation, and test directories with class subdirectories."""
    for split in ['train', 'validation', 'test']:
        for cls in CLASSES:
            path = dest_path / split / cls
            path.mkdir(parents=True, exist_ok=True)
    print("Destination directories created.")

def get_image_files(class_dir):
    """Retrieve all image files from a given class directory."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    return [file for file in class_dir.iterdir() if file.suffix.lower() in image_extensions]

def split_data(files, train_ratio, val_ratio, test_ratio):
    """Split files into train, validation, and test sets."""
    train_files, temp_files = train_test_split(
        files,
        train_size=train_ratio,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    validation_files, test_files = train_test_split(
        temp_files,
        train_size=val_ratio_adjusted,
        random_state=RANDOM_STATE,
        shuffle=True
    )
    return train_files, validation_files, test_files

def copy_files(files, destination):
    """Copy files to the destination directory."""
    for file in files:
        shutil.copy2(file, destination / file.name)

def organize_dataset():
    """Main function to organize the dataset."""
    # Step 1: Create destination directories
    create_directories(DEST_DIR)

    for cls in CLASSES:
        print(f"\nProcessing class: {cls}")
        class_source_dir = SOURCE_DIR / cls
        if not class_source_dir.exists():
            print(f"Source directory does not exist: {class_source_dir}")
            continue

        files = get_image_files(class_source_dir)
        print(f"Total files found for '{cls}': {len(files)}")

        if len(files) == 0:
            print(f"No image files found in {class_source_dir}. Skipping.")
            continue

        # Step 2: Split the data
        train_files, val_files, test_files = split_data(files, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO)
        print(f"Training: {len(train_files)} files")
        print(f"Validation: {len(val_files)} files")
        print(f"Testing: {len(test_files)} files")

        # Step 3: Copy the files to respective directories
        copy_files(train_files, DEST_DIR / 'train' / cls)
        copy_files(val_files, DEST_DIR / 'validation' / cls)
        copy_files(test_files, DEST_DIR / 'test' / cls)
    
    print("\nDataset organization complete.")

if __name__ == "__main__":
    organize_dataset()

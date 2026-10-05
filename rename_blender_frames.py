import os
import re

# Set DRY_RUN to True to preview changes without modifying files
DRY_RUN = False

# Matches either 3 or 4 digits, an underscore, and a suffix (e.g., 000_0.tif or 0000_0.tif)
PATTERN = re.compile(r"^(\d{3,4})_(\d+)\.tif$")


def select_directory():
    """Opens a graphical folder selection dialog if available."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()  # Hide main window
        root.attributes("-topmost", True)

        folder_selected = filedialog.askdirectory(title="Select Folder to Rename")
        return folder_selected
    except Exception:
        return None


def batch_rename(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        print("No valid directory selected. Exiting.")
        return

    print(f"Target Directory: {folder_path}\n" + "-" * 50)

    count = 0
    for filename in os.listdir(folder_path):
        match = PATTERN.match(filename)
        if match:
            num_part, suffix_part = match.groups()

            # Ensures the number part is padded to 4 digits
            # '000' -> '0000' | '0000' -> '0000'
            four_digit_num = num_part.zfill(4)
            new_filename = f"Blender_{four_digit_num}_{suffix_part}.tif"

            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, new_filename)

            if DRY_RUN:
                print(f"[PREVIEW] {filename}  -->  {new_filename}")
            else:
                os.rename(src, dst)
                print(f"[RENAMED] {filename}  -->  {new_filename}")

            count += 1

    print("-" * 50)
    print(f"Total files processed: {count}")


if __name__ == "__main__":
    folder_path = select_directory()

    if not folder_path:
        user_input = input("Enter directory path (or press Enter to exit): ").strip()
        folder_path = user_input.strip('"\'')

    if folder_path:
        batch_rename(folder_path)
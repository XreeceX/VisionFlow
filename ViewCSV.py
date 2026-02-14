"""
View and display traffic record images from TrafficRecords.csv.
Works in both Jupyter notebooks and standalone Python.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CSV_FILE = 'TrafficRecords.csv'
MAX_DISPLAY = 5


def display_frame(path: str, title: str | None = None) -> bool:
    """Display a single frame image. Returns True on success."""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return False
    try:
        img = mpimg.imread(path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title or os.path.basename(path))
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        print(f"Error displaying {path}: {e}")
        return False


def main():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()

    if 'FrameImage' not in df.columns:
        raise KeyError(
            f"Column 'FrameImage' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    for path in df['FrameImage'].head(MAX_DISPLAY):
        display_frame(path)


if __name__ == '__main__':
    main()

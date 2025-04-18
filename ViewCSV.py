import pandas as pd
import os
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

df = pd.read_csv('TrafficRecords.csv')
df.columns = df.columns.str.strip()
if 'FrameImage' not in df.columns:
    raise KeyError("Column 'FrameImage' not found in CSV. Please check your CSV file.")

for path in df['FrameImage'].head(5):
    if os.path.exists(path):
        try:
            display(Image(filename=path))
            img = mpimg.imread(path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(path.split("/")[-1])
            plt.show()
            
        except Exception as e:
            print(f"Error displaying {path}: {e}")
    else:
        print(f"File not found: {path}")
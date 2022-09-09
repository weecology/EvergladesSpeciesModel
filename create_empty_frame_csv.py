# Iterate through the empty folder and create a csv folder for model training/evaluation
from glob import glob
import pandas as pd
import os

images = glob("/blue/ewhite/everglades/empty-frames/**/*.png", recursive=True)
#images = [os.path.basename(x) for x in images]
df = pd.DataFrame({"image_path":images, "xmin":None,"ymin":None,"xmax":None,"ymax":None,"label":None})
df.to_csv("/blue/ewhite/everglades/Zooniverse/parsed_images/empty_frames.csv")
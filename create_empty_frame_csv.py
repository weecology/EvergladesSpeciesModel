# Iterate through the empty folder and create a csv folder for model training/evaluation
from glob import glob
import pandas as pd

images = glob("/blue/ewhite/everglades/empty-frames/**/*.png", recursive=True)
df = pd.DataFrame({"image_path":images, "xmin":None,"ymin":None,"xmax":None,"ymax":None,"label":None})
df.to_csv("/blue/ewhite/everglades/Zooniverse/parsed_images/empty_frames.csv")
import pandas as pd
import glob
import os
from deepforest import utilities

files = glob.glob("/Users/benweinstein/Dropbox/Weecology/everglades_species/test_LG/*.shp")
images = glob.glob("/Users/benweinstein/Dropbox/Weecology/everglades_species/test_LG/*.png")

dfs = []
for x in files:
    rgb = "/Users/benweinstein/Dropbox/Weecology/everglades_species/test_LG/{}.png".format(os.path.splitext(os.path.basename(x))[0])
    df = utilities.shapefile_to_annotations(shapefile=x, rgb=rgb)    
    df["image_path"] = "{}.png".format(os.path.splitext(os.path.basename(x))[0])
    dfs.append(df)

df = pd.concat(dfs)
df = df.reset_index(drop=True)
df.shape
df.label.value_counts()
df[df.label=="Unknown White"]
df.loc[df.label=="unknown White","label"] = "Unknown White"
df.loc[df.label=="Uknown White","label"] = "Unknown White"

df.loc[df.ymin < 0,"ymin"] = 0
df.loc[df.xmin < 0,"xmin"] = 0
df.loc[df.ymax > 1500,"ymax"] = 1500
df.loc[df.xmax > 1500,"xmax"] = 1500

df.to_csv("/Users/benweinstein/Dropbox/Weecology/everglades_species/test_LG/test.csv")
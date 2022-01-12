#Hard mining to look for rarer classes
from deepforest import main
import os
import cv2
import torch
import pandas as pd
import geopandas as gpd
from shapely import geometry

TRAINED_MODEL = "/blue/ewhite/everglades/Zooniverse/20211215_112228/species_model.pl"
train = pd.read_csv("/blue/ewhite/everglades/Zooniverse/parsed_images/species_train.csv")
label_dict = {key:value for value, key in enumerate(train.label.unique())}
m = main.deepforest(num_classes=len(label_dict), label_dict=label_dict)
m.load_state_dict(torch.load(TRAINED_MODEL, map_location="cpu")["state_dict"])
CROP_DIR = "/blue/ewhite/everglades/Zooniverse/mining/"
files = [
"/orange/ewhite/everglades/2021/SouthwestRanches/SouthwestRanches_04_19_2021_inspire.tif",
"/orange/ewhite/everglades/2021/SouthwestRanches/SouthwestRanches_04_19_2021_phantom.tif"
]

for f in files:
    basename = os.path.splitext(os.path.basename(f))[0]
    results = m.predict_tile(raster_path=f, mosaic=False)
    for index, result in enumerate(results):
        boxes, crop = result
        boxes = boxes[boxes.score > 0.5]
        filtered_boxes = boxes[boxes.label.isin(["Great Blue Heron","Snowy Egret","Roseate Spoonbill","Wood Stork"])] 
        highest_scores = filtered_boxes.groupby("label").apply(lambda x: x.sort_values(by="score",ascending=False).head(50)).reset_index(drop=True)
        if not highest_scores.empty:         
            highest_scores['geometry'] = highest_scores.apply(
                   lambda x: geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)            
            highest_scores = gpd.GeoDataFrame(highest_scores, geometry="geometry")
            highest_scores.to_file("{}/{}_{}.shp".format(CROP_DIR,basename, index))
            cv2.imwrite("{}/{}_{}.png".format(CROP_DIR,basename, index), crop)





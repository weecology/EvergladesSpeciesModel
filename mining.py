#Hard mining to look for rarer classes
from deepforest import main
import os
import cv2
import torch

TRAINED_MODEL = "/blue/ewhite/everglades/Zooniverse/20211215_112228/species_model.pl"
CROP_DIR = "/blue/ewhite/everglades/Zooniverse/mining/"
m = main.deepforest.load_from_checkpoint(TRAINED_MODEL)
m.model.load_state_dict(torch.load(TRAINED_MODEL))

files = [
"/orange/ewhite/everglades/2021/SouthwestRanches/SouthwestRanches_04_19_2021_inspire.tif",
"/orange/ewhite/everglades/2021/SouthwestRanches/SouthwestRanches_04_19_2021_phantom.tif"
]

for f in files:
    basename = os.path.splitext(os.path.basename(f))[0]
    results = m.predict_tile(raster_path=f, mosaic=False)
    for index, boxes, crop in enumerate(results):
        boxes = boxes[boxes.score > 0.5]
        filtered_boxes = boxes[boxes.label.isin("Great Blue Heron","Snowy Egret","Roseate Spoonbill","Wood Stork")] 
        highest_scores = filtered_boxes.groupby("label").apply(lambda x: x.sort_values(ascending=False).head(50))
        if not highest_scores.empty:
            highest_scores.to_file("{}/{}_{}.shp".format(CROP_DIR,basename, index))
            cv2.imwrite("{}/{}_{}.png".format(CROP_DIR,basename, index), crop)





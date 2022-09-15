#Hard mining to look for rarer classes
from deepforest import main
import os
import cv2
import geopandas as gpd
from shapely import geometry
import glob

TRAINED_MODEL = "/blue/ewhite/everglades/Zooniverse//20220910_182553/species_model.pl"
m = main.deepforest.load_from_checkpoint(TRAINED_MODEL)
CROP_DIR = "/blue/ewhite/everglades/Zooniverse/mining/"
files = glob.glob("/blue/ewhite/everglades/projected_mosaics/2022/*/*.tif")
files = [x for x in files if "Horus" in x]

for f in files:
    basename = os.path.splitext(os.path.basename(f))[0]
    results = m.predict_tile(raster_path=f, mosaic=False, patch_size=1500)
    if results is None:
        continue
    for index, result in enumerate(results):
        original_boxes, crop = result
        boxes = original_boxes[original_boxes.score > 0.4]
        filtered_boxes = boxes[boxes.label.isin(["Great Blue Heron","Snowy Egret","Roseate Spoonbill","Wood Stork"])] 
        highest_scores = filtered_boxes.groupby("label").apply(lambda x: x.sort_values(by="score",ascending=False).head(50)).reset_index(drop=True)
        if not highest_scores.empty:         
            original_boxes['geometry'] = original_boxes.apply(
                   lambda x: geometry.box(x.xmin, -x.ymin, x.xmax, -x.ymax), axis=1)            
            original_boxes = gpd.GeoDataFrame(original_boxes, geometry="geometry")
            original_boxes = original_boxes[original_boxes.score > 0.3]
            original_boxes.to_file("{}/{}_{}.shp".format(CROP_DIR,basename, index))
            crop = crop[:,:,::-1]
            cv2.imwrite("{}/{}_{}.png".format(CROP_DIR,basename, index), crop)


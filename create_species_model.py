#DeepForest bird detection from extracted Zooniverse predictions
import geopandas as gp
from shapely.geometry import Point, box
import pandas as pd
import rasterio
import os
import numpy as np
import glob
from datetime import datetime
import warnings
from pathlib import Path
import shutil

#Define shapefile utility
def shapefile_to_annotations(shapefile, rgb_path, buffer, savedir="."):
    """
    Convert a shapefile of annotations into annotations csv file for DeepForest training and evaluation
    Args:
        shapefile: Path to a shapefile on disk. If a label column is present, it will be used, else all labels are assumed to be "Tree"
        rgb_path: Path to the RGB image on disk
        savedir: Directory to save csv files
    Returns:
        None: a csv file is written
    """
    #Read shapefile
    gdf = gp.read_file(shapefile)
    
    #Drop any rounding errors duplicated
    gdf = gdf.groupby("selected_i").apply(lambda x: x.head(1))
    
    #define in image coordinates and buffer to create a box
    gdf["geometry"] =[Point(x,y) for x,y in zip(gdf.x.astype(float), gdf.y.astype(float))]
    gdf["geometry"] = [box(int(left), int(bottom), int(right), int(top)) for left, bottom, right, top in gdf.geometry.buffer(buffer).bounds.values]
        
    #extent bounds
    df = gdf.bounds
    
    #Assert size mantained
    assert df.shape[0] == gdf.shape[0]
    
    df = df.rename(columns={"minx":"xmin","miny":"ymin","maxx":"xmax","maxy":"ymax"})
    
    #cut off on borders
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')        
            
            with rasterio.open(rgb_path) as src:
                height, width = src.shape
    except:
        print("Image {} failed to open".format(rgb_path))
        os.remove(rgb_path)
        return None
    
    df.ymax[df.ymax > height] = height
    df.xmax[df.xmax > width] = width
    df.ymin[df.ymin < 0] = 0
    df.xmin[df.xmin < 0] = 0
    
    #add filename and bird labels
    df["image_path"] = os.path.basename(rgb_path)
    df["label"] = gdf["species"]
    
    #remove undesired classes
    df = df[~(df.label == "Unknown White Small")]
    
    #enforce pixel rounding
    df.xmin = df.xmin.astype(int)
    df.ymin = df.ymin.astype(int)
    df.xmax = df.xmax.astype(int)
    df.ymax = df.ymax.astype(int)
    
    #select columns
    result = df[["image_path","xmin","ymin","xmax","ymax","label"]]
     
    result = result.drop_duplicates()
    
    return result

def sample_if(x,n):
    """Sample up to n rows if rows is less than n
    Args:
        x: pandas object
        n: row minimum
        species_counts: number of each species in total data
    """
    if x.shape[0] < n:
        to_sample = n - x.shape[0]
        new_rows =  x.sample(to_sample, replace=True)
        return pd.concat([x, new_rows])
    else:
        return x

def find_rgb_path(shp_path, image_dir):
    basename = os.path.splitext(os.path.basename(shp_path))[0]
    rgb_path = "{}/{}.png".format(image_dir,basename)
    return rgb_path
    
def format_shapefiles(shp_dir, buffer, image_dir=None):
    """
    Format the shapefiles from extract.py into a list of annotations compliant with DeepForest -> [image_name, xmin,ymin,xmax,ymax,label]
    shp_dir: directory of shapefiles
    image_dir: directory of images. If not specified, set as shp_dir
    """
    if not image_dir:
        image_dir = shp_dir
        
    shapefiles = glob.glob(os.path.join(shp_dir,"*.shp"))
    
    #Assert all are unique
    assert len(shapefiles) == len(np.unique(shapefiles))
    
    annotations = [ ]
    for shapefile in shapefiles:
        rgb_path = find_rgb_path(shapefile, image_dir)
        result = shapefile_to_annotations(shapefile, rgb_path, buffer)
        #skip invalid files
        if result is None:
            continue
        annotations.append(result)
    annotations = pd.concat(annotations)
    
    return annotations

def split_test_train(annotations, resample_n=100):
    """Split annotation in train and test by image
    Args:
         annotations: dataframe of bounding box objects
         resample_n: resample classes under n images to n images
    """
    
    #Currently want to mantain the random split
    np.random.seed(0)
    
    #unique annotations for the bird detector
    #annotations = annotations.groupby("selected_i").apply(lambda x: x.head(1))
    
    #add to train_names until reach target split threshold
    image_names = annotations.image_path.unique()
    target = int(annotations.shape[0] * 0.92)
    counter = 0
    train_names = []
    for x in image_names:
        if target > counter:
            train_names.append(x)
            counter+=annotations[annotations.image_path == x].shape[0]
        else:
            break
        
    train = annotations[annotations.image_path.isin(train_names)]
    test = annotations[~(annotations.image_path.isin(train_names))]

    return train, test

def get_empty_frames(shp_dir, empty_frames_dir, max_empty_frames=0):
    empty_frames = glob.glob(f'{empty_frames_dir}/**/*.png')

    #Make sure the empty frames are in the main image directory
    shp_dir = Path(shp_dir)
    empty_frames_dir = Path(empty_frames_dir)
    for empty_frame in empty_frames:
        empty_frame = Path(empty_frame)
        if not Path.is_file(Path(shp_dir, empty_frame.name)):
            shutil.copy(empty_frame,
                        Path(shp_dir, empty_frame.name))

    #Add files with blank annotations
    empty_frames = [Path(path).name for path in empty_frames]
    empty_frames_df = pd.DataFrame(empty_frames, columns = ['image_path'])
    num_samples = min(max_empty_frames, len(empty_frames))
    assert num_samples > 0, "Empty frames were requested but no empty frames available in empty_frames_dir"
    print(f"Using {num_samples} empty frames split between train and test")
    empty_frames_df = empty_frames_df.sample(n=num_samples)
    empty_frames_df["xmin"] = 0
    empty_frames_df["ymin"] = 0
    empty_frames_df["xmax"] = 0
    empty_frames_df["ymax"] = 0
    empty_frames_df["label"] = "Empty"
    
    empty_train, empty_test = split_test_train(empty_frames_df)
    return empty_train, empty_test

def generate(shp_dir, empty_frames_dir=None, max_empty_frames=0, save_dir=".", buffer=25):
    """Parse annotations, create a test split and train a model"""
    print("[INFO] Generating data for species classification model")
    annotations = format_shapefiles(shp_dir, buffer)   
    
    #Split train and test
    print("[INFO] Generating test/train split")
    train, test = split_test_train(annotations, resample_n=1000)
    if empty_frames_dir and max_empty_frames > 0:
        print("[INFO] Adding empty frames")
        empty_train, empty_test = get_empty_frames(shp_dir, empty_frames_dir, max_empty_frames)
        train = pd.concat([train, empty_train])
        test = pd.concat([test, empty_test])
       
    #Enforce rounding to pixels, pandas "Int64" dtype for nullable arrays https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    train.xmin = train.xmin.astype("Int64")
    train.ymin = train.ymin.astype("Int64")
    train.xmax = train.xmax.astype("Int64")
    train.ymax = train.ymax.astype("Int64")
    
    test.xmin = test.xmin.astype("Int64")
    test.ymin = test.ymin.astype("Int64")
    test.xmax = test.xmax.astype("Int64")
    test.ymax = test.ymax.astype("Int64")
                
    #write paths to headerless files alongside data, add a seperate test empty file
    #Save
    
    train_path = "{}/species_train.csv".format(shp_dir)
    train.to_csv(train_path, index=False,header=True)
    test_path = "{}/species_test.csv".format(shp_dir)
    test.to_csv(test_path, index=False,header=True)
    if empty_frames_dir and max_empty_frames > 0:
        empty_test_path = "{}/empty_test.csv".format(shp_dir)
        empty_test.to_csv(empty_test_path, index=True)
    print("[INFO] Dataset generation complete")
        
    
if __name__ == "__main__":
    generate(
        shp_dir="/blue/ewhite/everglades/Zooniverse/parsed_images/",
        empty_frames_dir="/blue/ewhite/everglades/Zooniverse/parsed_images/empty_frames/",
        save_dir="/blue/ewhite/everglades/Zooniverse/predictions/"
    )
    

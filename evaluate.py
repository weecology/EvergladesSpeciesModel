#DeepForest bird detection from extracted Zooniverse predictions
import comet_ml
from pytorch_lightning.loggers import CometLogger
from deepforest import main
from empty_frames_utilities import *

import pandas as pd
import os
import numpy as np
import traceback
import torch
import tempfile

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path, PurePath
from datetime import datetime

def get_species_abbrev_lookup(species_lookup):
    species_abbrev_lookup = {}
    for number, species in species_lookup.items():
        split_name = species.split()
        abbrev = ''
        for sub_name in split_name:
            abbrev += sub_name[0]
        species_abbrev_lookup[number] = abbrev
    return species_abbrev_lookup

def index_to_example(index, results, test_path, comet_experiment):
    """Make example images of for confusion matrix"""
    tmpdir = tempfile.gettempdir()
    results = results.iloc[index]

    xmin = results['xmin']
    xmax = results['xmax']
    ymin = results['ymin']
    ymax = results['ymax']

    image_name = results['image_path']
    test_image_path = Path(test_path).parent
    image_path = PurePath(Path(test_image_path), Path(image_name))
    print(image_path)
    image = Image.open(str(image_path))

    draw = ImageDraw.Draw(image, "RGB")
    draw.rectangle((xmin, ymin, xmax, ymax), outline = (255, 255, 255), width=2)
    font = ImageFont.truetype("Gidole-Regular.ttf", 20)
    draw.text((xmin - 150, ymin - 150), f"image={image_name}\nxmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}", fill=(255, 255, 255), font=font)
    image = image.crop((xmin - 200, ymin - 200, xmax + 200, ymax + 200))

    tmp_image_name = f"{tmpdir}/confusion-matrix-{index}.png"
    image.save(tmp_image_name)

    results = comet_experiment.log_image(
        tmp_image_name, name=image_name,
    )
    plt.close("all")
    # Return sample, assetId (index is added automatically)
    return {"sample": tmp_image_name, "assetId": results["imageId"]}

def evaluate_model(test_path, model_path, empty_images_path=None, save_dir=".",
                experiment_name=None, comet_logger=None):
    """Evaluate a deepforest model
    Args:
        test_path: path to csv on disk, format image_path (relative to root dir), xmin, ymin, xmax, ymax, label
        model_path: a deepforest saved model
        empty_images_patH: path to csv on disk, same format as above, but with None, in xmin, ymin, xmax, ymax to denote empty frames
        experiment_name: used to name a new comet_experiment, superceded by comet_logger
        comet_logger: an existing comet logger to log metrics
    Returns:
        results: a pandas dataframe of deepforest results
    """ 
    if experiment_name is not None:
        comet_logger = CometLogger(project_name="everglades-species", workspace="weecology", experiment_name=experiment_name)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_savedir = "{}/{}".format(save_dir,timestamp)  
    
    try:
        os.mkdir(model_savedir)
    except Exception as e:
        print(e)
        
    test = pd.read_csv(test_path)

    ## Given that some test labels are unknown, seperately get box and class scores
    
    # Box Scores
    label_dict = {key:value for value, key in enumerate(test.label.unique())}    
    model = main.deepforest.load_from_checkpoint(model_path)
    original_label_dict = model.label_dict
    model.label_dict = label_dict
    model.numeric_to_label_dict = {value:key for key, value in label_dict.items()}
    box_score_results = model.evaluate(test_path, root_dir = os.path.dirname(test_path))
    
    # Class Scores
    test = test[test.label.isin(['Great Egret', 'Roseate Spoonbill', 'White Ibis',
           'Great Blue Heron', 'Wood Stork', 'Snowy Egret'])]
    
    test.to_csv("{}/cleaned_test_classes.csv".format(save_dir))
    
    model.label_dict = original_label_dict
    species_lookup = {value:key for key, value in original_label_dict.items()}
    model.numeric_to_label_dict = species_lookup      
    species_abbrev_lookup = get_species_abbrev_lookup(species_lookup)    
    
    results = model.evaluate("{}/cleaned_test_classes.csv".format(save_dir), root_dir = os.path.dirname(test_path))
    
    if comet_logger is not None:
        try:
            results["results"].to_csv("{}/iou_dataframe.csv".format(model_savedir))
            comet_logger.experiment.log_asset("{}/iou_dataframe.csv".format(model_savedir))

            results["predictions"].to_csv("{}/predictions_dataframe.csv".format(model_savedir))
            comet_logger.experiment.log_asset("{}/predictions_dataframe.csv".format(model_savedir))
            
            results["class_recall"].to_csv("{}/class_recall.csv".format(model_savedir))
            comet_logger.experiment.log_asset("{}/class_recall.csv".format(model_savedir))
            
            for index, row in results["class_recall"].iterrows():
                comet_logger.experiment.log_metric("{}_Recall".format(species_abbrev_lookup[row["label"]]),row["recall"])
                comet_logger.experiment.log_metric("{}_Precision".format(species_abbrev_lookup[row["label"]]),row["precision"])
            
            comet_logger.experiment.log_metric("Average Class Recall",results["class_recall"].recall.mean())
            comet_logger.experiment.log_metric("Box Recall",box_score_results["box_recall"])
            comet_logger.experiment.log_metric("Box Precision",box_score_results["box_precision"])
                        
            # Make predicted labels while dealing with test data that does not get a bounding box.
            # These predicted labels return as nan, so check for them using y == y (returns False for nan)
            # and then replace them with one more than the available class indexes for confusion matrix
            ypred = results["results"].predicted_label       
            ypred = np.asarray([model.label_dict[y] if y == y else model.num_classes for y in ypred])  
            ypred = torch.from_numpy(ypred)
            ypred = torch.nn.functional.one_hot(ypred.to(torch.int64), num_classes = model.num_classes + 1).numpy()
            
            # Code true labels to match indexes from model training
            ytrue = results["results"].true_label
            ytrue = np.asarray([model.label_dict[y] for y in ytrue])
            ytrue = torch.from_numpy(ytrue)

            # Create one hot representation with extra class for test data with no bounding box
            ytrue = torch.nn.functional.one_hot(ytrue.to(torch.int64), num_classes = model.num_classes + 1).numpy()

            # Add a label for undetected birds and create confusion matrix
            model.label_dict.update({'Bird Not Detected': 6})
            comet_logger.experiment.log_confusion_matrix(y_true=ytrue,
                                                         y_predicted=ypred,
                                                         labels = list(model.label_dict.keys()),
                                                         index_to_example_function=index_to_example,
                                                         results = results["results"],
                                                         test_path=test_path,
                                                         comet_experiment = comet_logger.experiment)
        except Exception as e:
            print("logger exception: {} with traceback \n {}".format(e, traceback.print_exc()))
    
    #Create a positive bird recall curve
    test_frame_df = pd.read_csv(test_path)
    dirname = os.path.dirname(test_path)
    test_frame_df["image_path"] = test_frame_df["image_path"].apply(lambda x: os.path.join(dirname,x))
    empty_images = test_frame_df.image_path.unique()  
    
    model.config["score_thresh"] = 0.01
    predict_empty_frames(model, empty_images, comet_logger, invert=True)
    
    #Test on empy frames
    if empty_images_path:
        #Set threshold very low to allow the creation of a precision recall curve
        model.config["score_thresh"] = 0.01        
        empty_frame_df = pd.read_csv(empty_images_path)
        empty_images = empty_frame_df.image_path.unique()    
        predict_empty_frames(model, empty_images, comet_logger)
        
        #Reset desired precision for uploading images
        model.config["score_thresh"] = 0.3        
        upload_empty_images(model, comet_logger, empty_images)

    return model

if __name__ == "__main__":
    evaluate_model(
                test_path="/blue/ewhite/everglades/Zooniverse/cleaned_test/test.csv",
                save_dir="/blue/ewhite/everglades/Zooniverse/",
                model_path="/blue/ewhite/everglades/Zooniverse//20220910_182547/species_model.pl",
                empty_images_path="/blue/ewhite/everglades/Zooniverse/parsed_images/empty_frames.csv",
                experiment_name="main")

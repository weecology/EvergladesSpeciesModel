#DeepForest bird detection from extracted Zooniverse predictions
import comet_ml
from pytorch_lightning.loggers import CometLogger
from deepforest import main
from deepforest import dataset
from deepforest import utilities
import create_species_model
from empty_frames_utilities import *
from evaluate import evaluate_model

import pandas as pd
import os
import numpy as np
import traceback
import torch
import tempfile

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path, PurePath
from pytorch_lightning import Trainer
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

def train_model(train_path, test_path, empty_images_path=None, save_dir=".",
                gbd_pretrain = True,
                experiment_name="ev-species",
                debug = False):
    """Train a DeepForest model"""
    
    comet_logger = CometLogger(project_name="everglades-species", workspace="bw4sz", experiment_name=experiment_name)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_savedir = "{}/{}".format(save_dir,timestamp)  
    tmpdir = tempfile.gettempdir()
    
    try:
        os.mkdir(model_savedir)
    except Exception as e:
        print(e)
    
    comet_logger.experiment.log_parameter("timestamp",timestamp)
    comet_logger.experiment.add_tag("species")
    
    # Log the number of training and test
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Add weak annotations from photoshop to train
    weak_train = pd.read_csv("/blue/ewhite/everglades/photoshop_annotations/split_annotations.csv")
    train = pd.concat([train, weak_train])
    train = train[train.label.isin(['Great Egret', 'Roseate Spoonbill', 'White Ibis',
           'Great Blue Heron', 'Wood Stork', 'Snowy Egret', 'Anhinga'])]
    test = test[test.label.isin(train.label)]
    
    # Add in weak annotations for empty frames
    empty_frames = pd.read_csv("/blue/ewhite/everglades/photoshop_annotations/inferred_empty_annotations.csv")
    empty_frames = empty_frames.sample(n=1000)
    empty_frames.image_path = empty_frames.image_path.apply(lambda x: os.path.basename(x))
    
    # Confirm no name overlaps
    overlapping_images = train[train.image_path.isin(empty_frames.image_path.unique())]
    if not len(overlapping_images) == 0:
        raise IOError("Overlapping images: {}".format(overlapping_images))
    
    train = pd.concat([train, empty_frames])
    
    # Store test train split for run to allow multiple simultaneous run starts
    train_path = str(PurePath(Path(train_path).parents[0], Path(f'species_train_{timestamp}.csv')))
    test_path = str(PurePath(Path(test_path).parents[0], Path(f'species_test_{timestamp}.csv')))
    train.to_csv(train_path)
    test.to_csv(test_path)

    comet_logger.experiment.log_table("train.csv", train)
    comet_logger.experiment.log_table("test.csv", test)

    # Set config and train    
    label_dict = {key:value for value, key in enumerate(train.label.unique())}
    species_lookup = {value:key for key, value in label_dict.items()}
    species_abbrev_lookup = get_species_abbrev_lookup(species_lookup)
    
    model = main.deepforest(num_classes=len(train.label.unique()),label_dict=label_dict)

    if gbd_pretrain:
    # Use the backbone and regression head from the global bird detector to transfer
    # learning about bird detection and bird related features 
        global_bird_detector = main.deepforest()
        global_bird_detector.use_bird_release()
        model.model.backbone.load_state_dict(global_bird_detector.model.backbone.state_dict())
        model.model.head.regression_head.load_state_dict(global_bird_detector.model.head.regression_head.state_dict())
    
    model.config["train"]["csv_file"] = train_path
    model.config["train"]["root_dir"] = os.path.dirname(train_path)
    
    # Set config and train
    model.config["validation"]["csv_file"] = test_path
    model.config["validation"]["root_dir"] = os.path.dirname(test_path)
    
    if debug:
        model.config["train"]["fast_dev_run"] = True
        model.config["gpus"] = None
        model.config["workers"] = 1
        model.config["batch_size"] = 1
        
    if comet_logger is not None:
        comet_logger.experiment.log_parameters(model.config)
        comet_logger.experiment.log_parameter("Training_Annotations",train.shape[0])    
        comet_logger.experiment.log_parameter("Testing_Annotations",test.shape[0])
        comet_logger.experiment.log_parameter("model_savedir",model_savedir)
    
    # Image callback significantly slows down training time, but can be helpful for debugging.
    # im_callback = images_callback(csv_file=model.config["validation"]["csv_file"], root_dir=model.config["validation"]["root_dir"], savedir=model_savedir, n=20)        
    
    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=model.config["gpus"],
        enable_checkpointing=False,
        max_epochs=model.config["train"]["epochs"],
        logger=comet_logger
    )
    
    ds = dataset.TreeDataset(csv_file=model.config["train"]["csv_file"],
                            root_dir=model.config["train"]["root_dir"],
                            transforms=dataset.get_transform(augment=True),
                            label_dict=model.label_dict)
    
    dataloader = torch.utils.data.DataLoader(ds,
                                        batch_size = model.config["batch_size"],
                                        collate_fn=utilities.collate_fn,
                                        num_workers=model.config["workers"])
    trainer.fit(model, dataloader)
    trainer.save_checkpoint("{}/species_model.pl".format(model_savedir))

    evaluate_model(test_path=test_path, model_path="{}/species_model.pl".format(model_savedir))
        
    return model
    

if __name__ == "__main__":
    regenerate = False
    max_empty_frames = 0
    if regenerate:
        print(f"[INFO] Regenerating dataset with up to {max_empty_frames} empty frames")
        create_species_model.generate(shp_dir="/blue/ewhite/everglades/Zooniverse/parsed_images/",
                                    empty_frames_path="/blue/ewhite/everglades/Zooniverse/parsed_images/empty_frames.csv",
                                    save_dir="/blue/ewhite/everglades/Zooniverse/predictions/",
                                    max_empty_frames=max_empty_frames,
                                    buffer=25)
    train_model(train_path="/blue/ewhite/everglades/Zooniverse/parsed_images/species_train.csv",
                test_path="/blue/ewhite/everglades/Zooniverse/cleaned_test/test.csv",
                save_dir="/blue/ewhite/everglades/Zooniverse/",
                gbd_pretrain=True,
                empty_images_path="/blue/ewhite/everglades/Zooniverse/parsed_images/empty_frames.csv",
                experiment_name="main")

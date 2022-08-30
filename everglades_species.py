#DeepForest bird detection from extracted Zooniverse predictions
import comet_ml
from pytorch_lightning.loggers import CometLogger
from deepforest.callbacks import images_callback
from deepforest import main
from deepforest import visualize
from deepforest import dataset
from deepforest import utilities
import pandas as pd
import os
import numpy as np
from datetime import datetime
import traceback
import torch
import tempfile
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path, PurePath
import torch.nn as nn
import math
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
import create_species_model
from pytorch_lightning import Trainer

def is_empty(precision_curve, threshold):
    precision_curve.score = precision_curve.score.astype(float)
    precision_curve = precision_curve[precision_curve.score > threshold]
    
    return precision_curve.empty

def empty_image(precision_curve, threshold):
    empty_true_positives = 0
    empty_false_negatives = 0
    for name, group in precision_curve.groupby('image'): 
        if is_empty(group, threshold):
            empty_true_positives +=1
        else:
            empty_false_negatives+=1
    empty_recall = empty_true_positives/float(empty_true_positives + empty_false_negatives)
    
    return empty_recall

def get_species_abbrev_lookup(species_lookup):
    species_abbrev_lookup = {}
    for number, species in species_lookup.items():
        split_name = species.split()
        abbrev = ''
        for sub_name in split_name:
            abbrev += sub_name[0]
        species_abbrev_lookup[number] = abbrev
    return species_abbrev_lookup

def plot_recall_curve(precision_curve, invert=False):
    """Plot recall at fixed interval 0:1"""
    recalls = {}
    for i in np.linspace(0,1,11):
        recalls[i] = empty_image(precision_curve=precision_curve, threshold=i)
    
    recalls = pd.DataFrame(list(recalls.items()), columns=["threshold","recall"])
    
    if invert:
        recalls["recall"] = 1 - recalls["recall"].astype(float)
    
    ax1 = recalls.plot.scatter("threshold","recall")
    
    return ax1
    
def predict_empty_frames(model, empty_images, comet_logger, invert=False):
    """Optionally read a set of empty frames and predict
        Args:
            invert: whether the recall should be relative to empty images (default) or non-empty images (1-value)"""
    
    #Create PR curve
    precision_curve = [ ]
    for path in empty_images:
        boxes = model.predict_image(path = path, return_plot=False)
        if boxes is not None:     
            boxes["image"] = path
            precision_curve.append(boxes)
    if len(precision_curve) == 0:
        return None
    
    precision_curve = pd.concat(precision_curve)
    recall_plot = plot_recall_curve(precision_curve, invert=invert)
    value = empty_image(precision_curve, threshold=0.4)
    
    if invert:
        value = 1 - value
        metric_name = "BirdRecall_at_0.4"
        recall_plot.set_title("Atleast One Bird Recall")
    else:
        metric_name = "EmptyRecall_at_0.4"
        recall_plot.set_title("Empty Recall")        
        
    comet_logger.experiment.log_metric(metric_name,value)
    comet_logger.experiment.log_figure(recall_plot)   
    
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
                balance_classes = False, balance_min = 0, balance_max = 100000,
                one_vs_all_sp = None,
                experiment_name="ev-species",
                debug = False):
    """Train a DeepForest model"""
    
    comet_logger = CometLogger(project_name="everglades-species", workspace="bw4sz", experiment_name=experiment_name)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_savedir = "{}/{}".format(save_dir,timestamp)  
    
    try:
        os.mkdir(model_savedir)
    except Exception as e:
        print(e)
    
    comet_logger.experiment.log_parameter("timestamp",timestamp)
    comet_logger.experiment.add_tag("species")
    
    #Log the number of training and test
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    #Add weak annotations from photoshop to train
    weak_train = pd.read_csv("/blue/ewhite/everglades/photoshop_annotations/split_annotations.csv")
    train = pd.concat([train, weak_train])
    
    if one_vs_all_sp:
        train["label"] = np.where(train["label"] == one_vs_all_sp, one_vs_all_sp, "Other Species")
        test["label"] = np.where(test["label"] == one_vs_all_sp, one_vs_all_sp, "Other Species")

    #Store test train split for run to allow multiple simultaneous run starts
    train_path = str(PurePath(Path(train_path).parents[0], Path(f'species_train_{timestamp}.csv')))
    test_path = str(PurePath(Path(test_path).parents[0], Path(f'species_test_{timestamp}.csv')))
    train.to_csv(train_path)
    test.to_csv(test_path)

    #Set config and train'    
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
    
    #Set config and train
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
        
    #im_callback = images_callback(csv_file=model.config["validation"]["csv_file"], root_dir=model.config["validation"]["root_dir"], savedir=model_savedir, n=20)        
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
    
    if balance_classes:
    #Overwrite sampler to weight by class
    
        #get class weights
        train_data = pd.read_csv(train_path)
        class_counts = train_data.groupby('label')['label'].count()
        class_counts[class_counts < balance_min] = balance_min    #Provide a floor to class weights
        class_counts[class_counts > balance_max] = balance_max    #Provide a ceiling to class weights
        class_weights = dict(class_counts / sum(class_counts))
        class_weights_numeric_label = {model.label_dict[key]: value for key, value in class_weights.items()}
        class_weights_numeric_label = {key: class_weights_numeric_label[key] for key in sorted(class_weights_numeric_label)}
    
        data_weights = []
        #upsample rare classes more as a residual
        for idx, batch in enumerate(ds):
            path, image, targets = batch
            labels = [model.numeric_to_label_dict[x] for x in targets["labels"].numpy()]
            image_weight = np.median([class_weights[x] for x in labels]) # mean or median instead of sum?
            data_weights.append(1 / image_weight)
            
        data_weights = data_weights / sum(data_weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights = torch.DoubleTensor(data_weights),
                                                                 num_samples=len(ds))
        dataloader = torch.utils.data.DataLoader(ds,
                                            batch_size = model.config["batch_size"],
                                            sampler = sampler,
                                            collate_fn=utilities.collate_fn,
                                            num_workers=model.config["workers"])
    else:
        dataloader = torch.utils.data.DataLoader(ds,
                                            batch_size = model.config["batch_size"],
                                            collate_fn=utilities.collate_fn,
                                            num_workers=model.config["workers"])

    # labs = []
    # for batch in dataloader:
    #     paths, x, y = batch
    #     batch_labels = np.concatenate([i["labels"].numpy() for i in y])
    #     labs.append(batch_labels)
    # labs = np.concatenate(labs)
    # pd.Series(labs).value_counts().sort_index() / sum(pd.Series(labs).value_counts())

    trainer.fit(model, dataloader)
    trainer.save_checkpoint("{}/species_model.pl".format(model_savedir))

    #Manually convert model
    results = model.evaluate(test_path, root_dir = os.path.dirname(test_path))
    
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
            comet_logger.experiment.log_metric("Box Recall",results["box_recall"])
            comet_logger.experiment.log_metric("Box Precision",results["box_precision"])
            
            comet_logger.experiment.log_parameter("saved_checkpoint","{}/species_model.pl".format(model_savedir))
            
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
    predict_empty_frames(model, empty_images, comet_logger, invert=True)
    
    #Test on empy frames
    if empty_images_path:
        empty_frame_df = pd.read_csv(empty_images_path)
        empty_images = empty_frame_df.image_path.unique()    
        predict_empty_frames(model, empty_images, comet_logger)

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
                test_path="/blue/ewhite/everglades/Zooniverse/parsed_images/species_test.csv",
                save_dir="/blue/ewhite/everglades/Zooniverse/",
                gbd_pretrain=True,
                balance_classes=False,
                balance_min = 1000,
                balance_max = 10000,
                one_vs_all_sp=None,
                experiment_name="longer_term_no_balance")

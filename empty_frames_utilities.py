import pandas as pd
import numpy as np
import cv2
import tempfile
import os

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
        else:
            boxes = pd.DataFrame({"image_path":[path], "xmin":[None],"ymin":[None],"xmax":[None],"ymax":[None],"label":[None],"image":[path]})
        precision_curve.append(boxes)
    
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

def upload_empty_images(model, comet_logger, empty_images):
    tmpdir = tempfile.gettempdir()    
    for x in empty_images:
        img = model.predict_image(path=x, return_plot=True)
        if img is not None:
            cv2.imwrite("{}/{}.png".format(tmpdir,os.path.basename(x)), img)
            comet_logger.experiment.log_image("{}/{}.png".format(tmpdir,os.path.basename(x)), image_scale=0.5)
        else:
            comet_logger.experiment.log_image(x, image_scale=0.5)    
#Test empty frames
from deepforest import main
from empty_frames_utilities import *
from pytorch_lightning.loggers import CometLogger
import pytest

@pytest.fixture()
def comet_logger():
    comet_logger = CometLogger(project_name="everglades-species", workspace="bw4sz", prefix="pytest")
    comet_logger.experiment.add_tag("pytest")

    return comet_logger

@pytest.fixture()
def model():
    model = main.deepforest()
    model.use_bird_release()
    
    return model

def test_predict_empty_frames(model, comet_logger):
    predict_empty_frames(model=model,
                         empty_images=["/Users/benweinstein/Documents/EvergladesSpeciesModel/tests/JetportSouth_03_08_2021_382.png"],
                         comet_logger=comet_logger)    
    
def test_upload_empty_images(model, comet_logger):
    upload_empty_images(model,
                        comet_logger,
                        empty_images=["/Users/benweinstein/Documents/EvergladesSpeciesModel/tests/JetportSouth_03_08_2021_382.png"])
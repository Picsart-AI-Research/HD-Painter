import torch
from segment_anything import sam_model_registry, SamPredictor
from .common import *

MODEL_PATH = f'{MODEL_FOLDER}/sam/sam_vit_h_4b8939.pth'
DOWNLOAD_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

# pre-download
# download_file(DOWNLOAD_URL, MODEL_PATH)


def load_model(device='cuda:0'):
    download_file(DOWNLOAD_URL, MODEL_PATH)
    sam = sam_model_registry["vit_h"](checkpoint=MODEL_PATH)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

import json
from os.path import join

import clip
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image

from utils import download_file


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(1.0),
            nn.Linear(1024, 128),
            nn.Dropout(1.0),
            nn.Linear(128, 64),
            nn.Dropout(1.0),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


DOWNLOAD_URL = 'https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/6934dd81792f086e613a121dbce43082cb8be85e/sac+logos+ava1-l14-linearMSE.pth'
MODEL_PATH = 'checkpoints/eval/aesthetic/sac+logos+ava1-l14-linearMSE.pth'

download_file(DOWNLOAD_URL, MODEL_PATH)

model = MLP(768)
model.load_state_dict(torch.load(MODEL_PATH))
model.to('cuda')
model.eval()

clip_model, clip_preprocess = clip.load('ViT-L/14', device='cuda')


def get_score(image: Image) -> float:
    image = clip_preprocess(image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        image_features = clip_model.encode_image(image).float()
    im_emb_arr = normalized(image_features.cpu().detach().numpy() )
    prediction = model(torch.from_numpy(im_emb_arr).cuda().type(torch.cuda.FloatTensor))
    return prediction.item()


if __name__ == '__main__':
    image = Image.open('sample_image.png')
    print(get_score(image))

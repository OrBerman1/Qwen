import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
pylab.rcParams['figure.figsize'] = 20, 12
from GLIP.maskrcnn_benchmark.config import cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)


config_file = "GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "GLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)

image = load('http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg')
caption = 'bobble heads on top of the shelf'
result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
imshow(result, caption)

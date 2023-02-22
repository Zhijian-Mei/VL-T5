from IPython.display import clear_output, Image, display
import PIL.Image
import io
import json
import torch
import numpy as np
from inference.processing_image import Preprocess
from inference.visualizing_image import SingleImageViz
from inference.modeling_frcnn import GeneralizedRCNN
from inference.utils import Config, get_data

import wget
import pickle
import os


URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

objids = get_data(OBJ_URL)
attrids = get_data(ATTR_URL)
gqa_answers = get_data(GQA_URL)
vqa_answers = get_data(VQA_URL)
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

# for visualizing output
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

image_filename = wget.download(URL)
image_dirname = image_filename
# frcnn_visualizer = SingleImageViz(image_filename, id2obj=objids, id2attr=attrids)

images, sizes, scales_yx = image_preprocess(image_filename)

output_dict = frcnn(
    images,
    sizes,
    scales_yx = scales_yx,
    padding = 'max_detections',
    max_detections = frcnn_cfg.max_detections,
    return_tensors = 'pt'
)

# add boxes and labels to the image
# frcnn_visualizer.draw_boxes(
#     output_dict.get("boxes"),
#     output_dict.get("obj_ids"),
#     output_dict.get("obj_probs"),
#     output_dict.get("attr_ids"),
#     output_dict.get("attr_probs"),
# )

# showarray(frcnn_visualizer._get_buffer())

normalized_boxes = output_dict.get("normalized_boxes")
features = output_dict.get("roi_features")

print(normalized_boxes)
print()
print(features)
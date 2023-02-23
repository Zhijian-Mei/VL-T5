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



directory = '../datasets/amazon_imgs'
counter = 0
for filename in os.listdir(directory):
    id = filename[:len(filename)-6]
    print(id)
    quit()
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        pass
image_filename = wget.download(URL)
image_dirname = image_filename
image_filenames = ['input.jpg','input (3).jpg']
# frcnn_visualizer = SingleImageViz(image_filename, id2obj=objids, id2attr=attrids)

images, sizes, scales_yx = image_preprocess(image_filenames)

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
print(normalized_boxes.shape)
print()
print(features)
print(features.shape)
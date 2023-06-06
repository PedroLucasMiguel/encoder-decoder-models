from lime import lime_image
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from models.densenet201_encoder_decoder.densenet import densenet201
import cv2
import warnings
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning) 

explainer = lime_image.LimeImageExplainer()

IMG_NAME = "images/dog4.png"


def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((224, 224)),
    ])    

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]) 
    return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

device = "cuda"
print(f"Using {device}")

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 
        
img = get_image(IMG_NAME)

model = densenet201(pretrained=torch.load("imagenet-training/densenet.pth"))
model.load_state_dict(torch.load("checkpoints/e_6_savestate.pt"))
model = nn.DataParallel(model)
model = model.to(device)

def classify_func(img):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in img), dim=0)
    batch = batch.to(device)
    outputs = model(batch)

    return F.softmax(outputs, dim=1).detach().cpu().numpy()


explanation = explainer.explain_instance(np.array(pill_transf(img)), classify_func, top_labels=1, hide_color=0, num_samples=1000)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
o_img = np.array(img)
print(img_boundry1.shape, mask.shape, o_img.shape, img_boundry1.shape)
img_boundry1 = cv2.resize(img_boundry1, dsize=(o_img.shape[1], o_img.shape[0]), interpolation=cv2.INTER_CUBIC)
plt.imshow(img_boundry1)
plt.show()
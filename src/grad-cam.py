from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from models.densenet201_encoder_decoder.densenet import densenet201
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

IMG_NAME = "images/cat-and-dog2.jpg"

device = "cpu"
print(f"Using {device}")

img = Image.open(IMG_NAME).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = preprocess(img)
input_batch = img.unsqueeze(0)
input_batch = input_batch.to(device)

model = densenet201(pretrained=torch.load("imagenet-training/densenet.pth"))
model.load_state_dict(torch.load("checkpoints/e_6_savestate.pt"))
model = model.to(device)

model.eval()

outputs = model(input_batch)
print(outputs)
outputs[:, 0].backward()

gradients = model.get_activations_gradient()
gradients = torch.mean(gradients, dim=[0, 2, 3])
layer_output = model.get_activations(input_batch)

for i in range(len(gradients)):
    layer_output[:, i, :, :] *= gradients[i]

layer_output = layer_output[0, : , : , :]

img = cv2.imread(IMG_NAME)

heatmap = torch.mean(layer_output, dim=0).detach().numpy()
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) 
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
cv2.imwrite("../output/gradient.jpg", heatmap)
superimposed_img = heatmap * 0.4 + img
final_img = np.concatenate((img, superimposed_img), axis=1)
cv2.imwrite("../output/map.jpg", final_img)
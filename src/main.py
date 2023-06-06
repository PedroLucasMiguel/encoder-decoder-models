from models.densenet201_encoder_decoder.densenet import densenet201
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import torch
from matplotlib import pyplot as plt
import numpy as np

preprocess = transforms.Compose([
    transforms.Resize((224, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("images/dog3.jpg")

img = preprocess(img).unsqueeze(0).to("cuda")

model = densenet201(pretrained=torch.load("imagenet-training/densenet.pth")).to("cuda")
model.load_state_dict(torch.load("checkpoints/e_6_savestate.pt"))

pred = model.process_with_encoder_decoder(img)
print(pred)
pred = pred.detach().cpu().numpy()
pred = pred[0, :, :, :]
pred = pred.reshape((224, 244, 3))
pred = (pred / (pred + 1))* 255
#pred = ((pred + 1) / 2) * 255
pred = np.round(pred).astype(np.uint8)
plt.imshow(pred)
plt.show()

    
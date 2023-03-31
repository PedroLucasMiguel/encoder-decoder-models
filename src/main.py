from models.densenet201_encoder_decoder.densenet import densenet201
from torchvision import transforms
from PIL import Image
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 244)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("images/gatos.jpg")

img = preprocess(img).unsqueeze(0).to("cuda")

model = densenet201(pretrained=torch.load("imagenet-training/densenet.pth")).to("cuda")
model.load_state_dict(torch.load("checkpoints/e_6_savestate.pt"))

print(model)

#image = activation["decoder"].detach().cpu().numpy()
#image = image[0, :, :, :]
#image = image.reshape((224, 244, 3))
#image = image + 1.0
#image = image * 127.5
#image = np.round(image).astype(np.uint8)
#plt.imshow(image[:,:, 0])
#plt.show()

#print(model)
#for name, params in model.named_parameters():
#    if params.requires_grad == True:
#            print(name)

#with torch.no_grad():
#    output = model(img)
#    a = output.detach().cpu().numpy()
#    a *= (255.0/a.max())
#    print(a)
#    cv2.imwrite("a.png", a)

    
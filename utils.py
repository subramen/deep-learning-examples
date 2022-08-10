import torch
import os
import requests
import ast
from torchvision import transforms

def download_image(url):
    if url.startswith("http"):
        r = requests.get(url, stream=True).content
        open("input.jpg", "wb").write(r)
        url = "input.jpg"
    return url


def get_mapping_dict(idx_to_labels_url):
    # idx_to_labels_url = "https://gist.githubusercontent.com/suraj813/1fe4c9dd0bc7e1dd1ce79462712ac9ce/raw/0e2c65813946769a375d673a34a1c0236b0505f1/coco_idx_to_labels.txt"
    r = requests.get(idx_to_labels_url).text
    return ast.literal_eval(r)


def apply_imagenet_transform(img):
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    return transform(img)


# def print_sizeof(model):
#     total = 0
#     for p in model.parameters():
#         total += p.numel() * p.element_size()
#     total /= 1e6
#     print("Model size: ", total, " MB")


def print_size_of_model(model):
    torch.jit.script(model).save("temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
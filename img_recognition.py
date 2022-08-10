import torch
from torchvision import models, transforms
import requests
import ast


# get the Imagenet {class: label} mapping
CLS_IDX = requests.get("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
CLS_IDX = ast.literal_eval(CLS_IDX.text)

def load_img(url):
    from PIL import Image
    if url.startswith("https"):
        img = Image.open(requests.get(url, stream=True).raw)
    else:
        img = Image.open(url)
    return img


def preprocess(img):
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    img = transform(img).unsqueeze(0)
    return img


def load_model():
    model = models.mobilenet_v3_large(pretrained=True)
    model.eval().requires_grad_(False)
    return model


def get_predictions(outp):
    outp = torch.nn.functional.softmax(outp, dim=1)
    score, idx = torch.topk(outp, 1)
    idx.squeeze_()
    predicted_label = CLS_IDX[idx.item()]
    print(predicted_label, '(', score.squeeze().item(), ')')


if __name__ == "__main__":
    # Download remote image
    img = load_img("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png")
    img = preprocess(img)

    # Load model
    model = load_model()

    # Get model output and human text prediction
    logits = model(img)
    get_predictions(logits)
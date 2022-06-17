# torch==1.10.1
# torchvision==0.11.2
# pillow==9.0.1

from PIL import Image
from torchvision import models, transforms as T
import utils
from pprint import pprint
from collections import Counter

def load_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model


def load_input(img_path):
    image = Image.open(img_path)
    image = T.ToTensor()(image)
    return [image]


def get_obj_counts(model_output):
    _, labels, scores = model_output[0].values()
    label_map = utils.get_mapping_dict()
    detected_objects = []
    
    # filter out low-confidence predictions
    confidence_threshold = 0.85
    for label, score in zip(labels.tolist(), scores.tolist()):
        if score > confidence_threshold:
            classname = label_map[str(label)]
            detected_objects.append((classname, score,))
    
    counts = Counter([x[0] for x in detected_objects])
    return detected_objects, counts 
    

def main(img_path):
    # show the image we're passing to the model
    img_path = utils.download_image(img_path)
    Image.open(img_path).show()

    model = load_model()                # Load pretrained torchvision model
    X = load_input(img_path)            # Load image as tensor
    predictions = model(X)              # Run inference
    detected_objects, counts = get_obj_counts(predictions)   

    print("Detected objects:")
    print("="*20)
    pprint(detected_objects)
    print()
    
    print("Count of objects:")
    print("="*20)
    pprint(counts)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])

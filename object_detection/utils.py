import requests
import ast

def download_image(url):
    if url.startswith("http"):
        r = requests.get(url).content
        open("input.jpg", "wb").write(r)
        url = "input.jpg"
    return url

def get_mapping_dict():
    idx_to_labels_url = "https://gist.githubusercontent.com/suraj813/1fe4c9dd0bc7e1dd1ce79462712ac9ce/raw/0e2c65813946769a375d673a34a1c0236b0505f1/coco_idx_to_labels.txt"
    r = requests.get(idx_to_labels_url).text
    return ast.literal_eval(r)
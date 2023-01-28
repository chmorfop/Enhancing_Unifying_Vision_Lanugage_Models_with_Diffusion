import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse

def main(clip_model_type: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_trainy.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./data/coco/annotations/train_caption.json', 'r') as f:
        data1 = json.load(f)
    data = data1['annotations']
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    counter = 0
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            # run the forward pass of the model to get the image features.
            # result 1x512
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)

        # example {'image_id': 203564, 'id': 37, 'caption': 'A bicycle replica with a clock as the front wheel.', 'clip_embedding': 0}
        all_captions.append(d)
        if counter == 2:
            break
        counter = counter + 1
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
    # print(torch.cat(all_embeddings, dim=0).size())
    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))

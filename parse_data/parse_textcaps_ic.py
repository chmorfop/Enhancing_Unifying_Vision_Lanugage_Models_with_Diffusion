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
    print('device : {}'.format(device))
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/textcaps/clip_feat_{clip_model_name}_train_ic.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    annotation_path = './data/textcaps/train.json'
    with open(annotation_path, 'r') as f:
        ann = json.load(f).get('data')

    print("# %0d QAs loaded from json " % len(ann))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(ann))):
        temp_ann_img_id = ann[i].get('image_id')
        temp_image_pth = ann[i].get('image_path')
        temp_ann_caption = ann[i].get('caption_str')
        prepath = ".data/textcaps/"
        try:
            image = io.imread(prepath+temp_image_pth)
        except Exception as e:
            print(i, temp_ann_img_id)
            print(e)
            raise
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        temp_dict = {
                     'caption': temp_ann_caption,
                     'clip_embedding': i,
                     'image_id': temp_ann_img_id,
                     }
        all_embeddings.append(prefix)
        all_captions.append(temp_dict)
    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
    print("%0d embeddings saved." % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
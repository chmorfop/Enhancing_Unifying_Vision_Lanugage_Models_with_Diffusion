import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from diffusers import StableDiffusionPipeline



def main(clip_model_type: str):

    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.safety_checker = lambda images, clip_input: (images, False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : {}'.format(device))
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/diffgen_clip_feat_{clip_model_name}_train_ic.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    annotation_path = './data/coco/annotations/captions_train2014.json'
    with open(annotation_path, 'r') as f:
        ann = json.load(f).get('annotations')

    print("# %0d QAs loaded from json " % len(ann))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(ann))):
        temp_ann_img_id = ann[i].get('image_id')
        temp_ann_caption = ann[i].get('caption')

        image = pipe(temp_ann_caption).images[0]
        image.save('./data/coco/generative_images/{}.jpg'.format(i))
        image = preprocess(image).unsqueeze(0).to(device)

        # 414113
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

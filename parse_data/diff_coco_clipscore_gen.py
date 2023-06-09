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


def decapitalize_first_letter(s, upper_rest=False):
    return ''.join([s[:1].lower(), (s[1:].upper() if upper_rest else s[1:])])


def get_clip_score(image, text, clip_model, preprocess):
    # Load the pre-trained CLIP model and the image
    # model, preprocess = clip.load('ViT-B/32')

    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])

    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    # model = model.to(device)

    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)

    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()

    return clip_score


def find_best_clip_score(gen_images, paraphrased_texts, clip_model, preprocess):
    max_clip_score = -1
    index_clip_score = None
    for i, (temp_im, temp_txt) in enumerate(zip(gen_images, paraphrased_texts)):
        temp = get_clip_score(temp_im, temp_txt, clip_model, preprocess)
        if max_clip_score < temp:
            max_clip_score = temp
            index_clip_score = i
    return index_clip_score


def main(clip_model_type: str):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.safety_checker = lambda images, clip_input: (images, False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : {}'.format(device))
    clip_model_name = clip_model_type.replace('/', '_')
    # TODO
    out_path = f"./data/coco/batch_7_diffgen_clipscore_clip_feat_{clip_model_name}_train_ic.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    annotation_path = './data/coco/annotations/captions_train2014.json'
    with open(annotation_path, 'r') as f:
        ann = json.load(f).get('annotations')

    print("# %0d QAs loaded from json " % len(ann))
    all_embeddings = []
    all_captions = []
    # TODO
    for i in tqdm(range(87500,100000)):
        temp_ann_img_id = ann[i].get('image_id')
        temp_ann_caption = ann[i].get('caption')

        tempy = 'High photo-realistic, ' + decapitalize_first_letter(temp_ann_caption)
        tempy_default_list = [temp_ann_caption] * 5
        temp_gen_images = pipe([tempy] * 5, num_inference_steps=100, guidance_scale=7.5).images

        res_index = find_best_clip_score(temp_gen_images, tempy_default_list, clip_model, preprocess)
        image = temp_gen_images[res_index]

        image.save('./data/coco/generative_images_clipscore/{}.jpg'.format(i))
        image = preprocess(image).unsqueeze(0).to(device)

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

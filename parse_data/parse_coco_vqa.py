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
    out_path = f"/content/clipcap/data/coco/clip_feat_{clip_model_name}_train_vqa.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    annotation_path = '/content/drive/MyDrive/Colab Notebooks/COCO/Annotations/VQA_train2014_annotations.json'
    question_path = '/content/drive/MyDrive/Colab Notebooks/COCO/Annotations/VQA_train2014_questions.json'

    with open(annotation_path, 'r') as f:
        ann = json.load(f).get('annotations')

    with open(question_path, 'r') as f:
        questions = json.load(f).get('questions')

    print("# %0d QAs loaded from json " % len(ann))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(ann))):
        temp_ann_question_id = ann[i].get('question_id')
        temp_q_question_id = questions[i].get('question_id')
        temp_ann_img_id = ann[i].get('image_id')
        temp_q_img_id = questions[i].get('image_id')
        if (temp_ann_question_id != temp_q_question_id) and (temp_ann_img_id != temp_q_img_id):
            raise
        temp_question = questions[i]
        temp_ann = ann[i]
        img_id = temp_ann["image_id"]
        prepath = "/content/output"
        filename = f"/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        try:
            image = io.imread(prepath+filename)
        except Exception as e:
            print(i, img_id)
            print(e)
            raise
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        temp_dict = {'question': temp_question.get('question'),
                     'answer': temp_ann.get('multiple_choice_answer'),
                     'clip_embedding': i,
                     'image_id': temp_ann.get('image_id'),
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
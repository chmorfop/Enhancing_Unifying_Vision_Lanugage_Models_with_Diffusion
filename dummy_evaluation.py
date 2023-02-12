from evaluation.bleu.bleu import Bleu
from evaluation.rouge.rouge import Rouge
from evaluation.cider.cider import Cider
from evaluation.meteor.meteor import Meteor
from evaluation.tokenizer.ptbtokenizer import PTBTokenizer
import json

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


if __name__ == '__main__':

    # Example ..
    # gen = {}
    # gts = {}
    # caps_gt = [
    #     'A train traveling down tracks next to lights.',
    #     'A blue and silver train next to train station and trees.',
    #     'A blue train is next to a sidewalk on the rails.',
    #     'A passenger train pulls into a train station.',
    #     'A train coming down the tracks arriving at a station.']

    # caps_gen = ['train traveling down a track in front of a road']

    # gen['0'] = caps_gen
    # gts['0'] = caps_gt

    temp_path = '/content/drive/MyDrive/Colab Notebooks/COCO/Results/full_gt_dict_vqa.json'

    with open(temp_path) as json_file:
        data = json.load(json_file)

    gen = {}
    gts = {}
    for k in data.keys() :
      temp = data[k]
      gen[k] = [temp.get('predicted_answer')]
      gts[k] = [temp.get('answer')]

    gts = PTBTokenizer.tokenize(gts)
    gen = PTBTokenizer.tokenize(gen)

    print(score(gts, gen))
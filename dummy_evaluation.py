from evaluation.bleu.bleu import Bleu
from evaluation.rouge.rouge import Rouge
from evaluation.cider.cider import Cider
from evaluation.meteor.meteor import Meteor
from evaluation.tokenizer.ptbtokenizer import PTBTokenizer


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
    gen = {}
    gts = {}
    caps_gt = [
        'A train traveling down tracks next to lights.',
        'A blue and silver train next to train station and trees.',
        'A blue train is next to a sidewalk on the rails.',
        'A passenger train pulls into a train station.',
        'A train coming down the tracks arriving at a station.']

    caps_gen = ['train traveling down a track in front of a road']

    caps_gty = ['A narrow kitchen filled with appliances and cooking utensils.',
                'A galley kitchen with cabinets and appliances on both sides',
                'A hallway leading into a white kitchen with appliances.',
                'Doorway view of a kitchen with a sink, stove, refrigerator and pantry.',
                'The pantry door of the small kitchen is closed.']

    caps_geny = ['A narrow kitchen with appliances ']

    gen['0'] = caps_gen
    gts['0'] = caps_gt

    gts = PTBTokenizer.tokenize(gts)
    gen = PTBTokenizer.tokenize(gen)

    print(score(gts, gen))
# import matplotlib.pyplot as plt
#
# epochs = [i for i in range(20)]
# print(epochs)
# epoch_train_loss = [i for i in range(20)]
#
# fig, axes = plt.subplots(1 , figsize=(15, 15))
#
# plt.plot(epochs, epoch_train_loss, color='b', linestyle='-', label='Training loss')
# plt.title('Training Loss & Epochs', fontsize=16)
# plt.xlabel('Epochs', fontsize=16)
# plt.ylabel('Loss', fontsize=16)
# plt.legend()
# plt.savefig('./epoch_train_loss')

import pickle

from tqdm import tqdm
#
# with open('data/coco/oscar_split_ViT-B_32_trainy_vqa_1024.pkl', 'rb') as f:
#     all_data = pickle.load(f)
#
# for epoch in range(2):
#     for d in tqdm(all_data,total=len(all_data),desc='Epoch ' + str(epoch)):
#         print('..')

#
# print
# 'tokenization...'
# tokenizer = PTBTokenizer()
# gts = tokenizer.tokenize(gts)
# res = tokenizer.tokenize(res)
#
# # =================================================
# # Set up scorers
# # =================================================
# print
# 'setting up scorers...'
# scorers = [
#     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
#     (Meteor(), "METEOR"),
#     (Rouge(), "ROUGE_L"),
#     (Cider(), "CIDEr"),
#     (Spice(), "SPICE")
# ]
#
# # =================================================
# # Compute scores
# # =================================================
# for scorer, method in scorers:
#     print
#     'computing %s score...' % (scorer.method())
#     score, scores = scorer.compute_score(gts, res)
#     if type(method) == list:
#         for sc, scs, m in zip(score, scores, method):
#             self.setEval(sc, m)
#             self.setImgToEvalImgs(scs, gts.keys(), m)
#             print
#             "%s: %0.3f" % (m, sc)
#     else:
#         self.setEval(score, method)
#         self.setImgToEvalImgs(scores, gts.keys(), method)
#         print
#         "%s: %0.3f" % (method, score)

import json

# Opening JSON file
# f = open('/home/chris/PycharmProjects/CLIP_prefix_caption/data/VQA/val_data.json')
f = open('/media/chris/4f8d85a4-7412-4e22-89be-f483a57450c0/home/morf/Desktop/val_data.json')

# returns JSON object as
# a dictionary
# data = json.load(f)
# # print(len(data))
# print()
# # print(data[:1])
# # tt = 'A child holding a flowered umbrella and petting a yak'
# # 403013
# zz = [['A child holding a flowered umbrella and petting a yak.', 'A young man holding an umbrella next to a herd of cattle.', 'a young boy barefoot holding an umbrella touching the horn of a cow', 'A young boy with an umbrella who is touching the horn of a cow.', 'A boy holding an umbrella while standing next to livestock.'], ['A narrow kitchen filled with appliances and cooking utensils.', 'A galley kitchen with cabinets and appliances on both sides', 'A hallway leading into a white kitchen with appliances.', 'Doorway view of a kitchen with a sink, stove, refrigerator and pantry.', 'The pantry door of the small kitchen is closed.'], ['A little girl holding a kitten next to a blue fence.', 'Girl in a tank top holding a kitten in her back yard', 'A young girl is holding a small cat.', 'Girl with a yellow shirt holding a small cat', 'A girl smiles as she holds a kitty cat.'], ['A toilet sitting in a bathroom next to a sink.', 'A toilet in a bathroom with green faded paint.', 'A bathroom with a gouged wall and wall socket with no panel. ', 'The toilet is near the door in the bathroom.', 'The bathroom wall needs to be resurfaced and painted.'], ['There are two sinks next to two mirrors.', 'Two very metallic sinks are shown as well as the mirrors above them.', 'A room with stainless steel equipment including sinks and mirrors.', 'Two stainless steel sinks with mirrors and a fire extinguisher. ', 'A photo of a bathroom made of steel.'], ['A woman rides a bicycle on a road next to the median.', 'A girl is riding her bike down the street.', 'A lady riding her bicycle on the side of a street.', 'A person on a bike riding on a street.', 'A woman riding a bike down a street next to a divider.'], ['A bath tub sitting next to a sink in a bathroom.', 'The bathroom is clean and ready for us to use.', 'a bath tub sitting next to a sink ', 'A bathroom that has a shower curtain over a bathtub.', 'A bathroom with multicolored tile, bathtub and pedestal sink.'], ['A row of parked cars sitting next to parking meters.', 'A row of cars parked on a street with parking meters.', 'A series of parking meters and cars are located next to each other. ', 'A parking meter on a street by a car with traffic.', 'A parking meter on a street with cars'], ['An elegant bathroom features a tub, sink, mirror, and decorations. ', 'An old fashion above ground tub is shown with gold feet.', 'A lovely, vintage-styled bathroom with a great claw-footed tub. ', 'Bathroom with a pedestal sink and claw foot bathtub', 'A claw foot tub is in a large bathroom near a pedestal sink.'], ['A red helmet is on a yellow toilet in the dirt.', 'A yellow toilet with a red helmet on top of it.', 'A yellow toilet with a red helmet above it.', 'A toilet sitting in an outdoor area with a helmet resting on top of it.', 'An old toilet sits in dirt with a helmet on top.']]
# for d in data:
#     temp = d.get('caption')
#     for z in zz:
#         if temp in z:
#             print(d.get('image_id'))
#             print(d.get('caption'))
#             print(z)

import itertools

gen = {}
gts = {}
it = 0

caps_gt = [['A narrow kitchen filled with appliances and cooking utensils.',
            'A galley kitchen with cabinets and appliances on both sides',
            'A hallway leading into a white kitchen with appliances.',
            'Doorway view of a kitchen with a sink, stove, refrigerator and pantry.',
            'The pantry door of the small kitchen is closed.']]

caps_gen = [['A simple kitchen with appliances and a white kitchen board']]
for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
    print(gts_i)
    print(gen_i)
    ll= []
    for k,g in itertools.groupby(gen_i):
        print(k)
        ll.append(k)
    print(ll)
    print(' '.join([k for k, g in itertools.groupby(gen_i)]))
    # gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
    # gen['%d_%d' % (it, i)] = [gen_i, ]
    # gts['%d_%d' % (it, i)] = gts_i

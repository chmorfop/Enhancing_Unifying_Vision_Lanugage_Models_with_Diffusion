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
# f = open('/media/chris/4f8d85a4-7412-4e22-89be-f483a57450c0/home/morf/Desktop/val_data.json')

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

# import itertools
#
# gen = {}
# gts = {}
# it = 0
#
# caps_gt = [['A narrow kitchen filled with appliances and cooking utensils.',
#             'A galley kitchen with cabinets and appliances on both sides',
#             'A hallway leading into a white kitchen with appliances.',
#             'Doorway view of a kitchen with a sink, stove, refrigerator and pantry.',
#             'The pantry door of the small kitchen is closed.']]
#
# caps_gen = [['A simple kitchen with appliances and a white kitchen board']]
# for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
#     print(gts_i)
#     print(gen_i)
#     ll= []
#     for k,g in itertools.groupby(gen_i):
#         print(k)
#         ll.append(k)
#     print(ll)
#     print(' '.join([k for k, g in itertools.groupby(gen_i)]))
#     # gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
#     # gen['%d_%d' % (it, i)] = [gen_i, ]
#     # gts['%d_%d' % (it, i)] = gts_i



# import torch
#
# # Initialize a 3D tensor with random values
# a = torch.rand(3, 4, 5)
# print("Original tensor:")
# print(a)
# print(a.size())
#
# # Define the threshold value
# threshold = 0.5
#
# # Create a Boolean mask to select only the values below the threshold
# mask = a < threshold
# print("Boolean mask:")
# print(mask)
# print(mask.size())
#
# # Use the Boolean mask to index into the tensor and select only the values below the threshold
# filtered = a[mask]
# print("Filtered tensor:")
# print(filtered.size())
# print()
#
# print(filtered)

# import torch
#
# # Initialize a 3D tensor with random values
# a = torch.rand(3, 4, 5)
# print("Original tensor:")
# print(a)
# print(a.size())
#
# # Define the threshold value
# threshold = 0.5
#
# # Create a Boolean mask to select only the values above the threshold
# mask = a > threshold
#
# # Use the Boolean mask to fill the values above the threshold with -inf
# a.masked_fill_(mask, float("-inf"))
# print("Masked tensor:")
# print(a.size())
# print()
# print(a)


# sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
# sorted_indices_to_remove[..., 0] = 0

# import torch
#
# array = torch.rand( 3, 10)
#
# print(array)
# print(array.size())
# print()
# print('***')
# print()
#
# # kovei to prwto stoixeio
# # print(array[...,1:])
# # print(array[...,1:].size())
# # print()
# # kovei to teleutaio stoixeio
# # print(array[...,:-1])
# # print(array[...,:-1].size())
# # print()
# # to prwto stoixeio to kanei 0?
# # array[..., 0] = 0
#
#
# array[..., 1:] = array[..., :-1].clone()
# array[..., 0] = 0
# print(array)
# print(array.size())


import torch


# # Create a tensor of 5 elements
# array = torch.rand( 3, 10)
# print(array)
#
# sorted_logits, sorted_indices = torch.sort(array, descending=True)
#
# # Create a boolean mask with the same size as the tensor
# mask = sorted_logits > 0.2
# print(mask)
#
#
# # The result is a tensor with only the elements from the original tensor that satisfy the condition
# # print()
# # print()
# # print(result) # tensor([3, 4, 5])
# # print(result.size())
#
#
# tempy = array * mask
# print(tempy)
# print(tempy.size())
#
# tempy[:, torch.tensor(2,3)]=10
# print(tempy)

# import torch
#
# # Initialize a 3D tensor with random values
# # a = torch.rand(2, 2, 2)
# a = torch.tensor([1, 2, 3])
# print(a)
# print(a.all())


# import torch
#
#
# # Create a tensor of 5 elements
# array = torch.rand( 3, 10)
# print(array)
#
# testy = torch.tensor([[ 2,4],
#         [2,4],
#         [2,4]])
# print(testy)
# print(testy.size())
# array[:, testy] = -float("Inf")
# print(array)


# import torch
#
# basic_tensor = torch.randn(3, 50257)
# boolean_tensor = (basic_tensor > 0).to(torch.float)
# print(boolean_tensor)
# masked_tensor = torch.masked_select(basic_tensor, boolean_tensor.bool())
# masked_tensor = masked_tensor.reshape( -1)
# print(masked_tensor)

gg = [203564, 179765, 322141, 16977, 106140, 106140, 322141, 322141, 322141, 203564, 179765, 16977, 106140, 322141, 571635, 301837, 190236, 315702, 331352, 517069, 315702, 189634, 472598, 162113, 203564, 179765, 331352, 126657, 285421, 16977]
zz = ['A bicycle replica with a clock as the front wheel.', 'A black Honda motorcycle parked in front of a garage.', 'A room with blue walls and a white sink and door.', 'A car that seems to be parked illegally behind a legally parked car', 'A large passenger airplane flying through the air.', 'There is a GOL plane taking off in a partly cloudy sky.', 'Blue and white color scheme in a small bathroom.', 'This is a blue and white bathroom with a wall sink and a lifesaver on the wall.', 'A blue boat themed bathroom with a life preserver on the wall', 'The bike has a clock as a tire.', 'A Honda motorcycle parked in a grass driveway', 'two cars parked on the sidewalk on the street', 'An airplane that is, either, landing or just taking off.', 'A bathroom with walls that are painted baby blue.', 'A bathroom with a toilet, sink, and shower.', 'A long empty, minimal modern skylit home kitchen.', 'An office cubicle with four different types of computers.', 'A bathroom sink with toiletries on the counter.', 'A small closed toilet in a cramped space.', 'Two women waiting at a bench next to a street.', 'A bathroom sink and various personal hygiene items.', 'This is an open box containing four cucumbers.', 'An old-fashioned green station wagon is parked on a shady driveway.', 'A gas stove next to a stainless steel kitchen sink and countertop.', 'A black metal bicycle with a clock inside the front wheel.', 'A black Honda motorcycle with a dark burgundy seat.', 'A tan toilet and sink combination in a small room.', 'Several motorcycles riding down the road in formation.', 'A black cat is inside a white toilet.', 'City street with parked cars and a bench.']


def merge(list1, list2):
    assert len(list1) == len(list2)
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

mrg = merge(gg, zz)


temp_dict = {}
for key, value in mrg:
    if key in temp_dict:
        temp_dict[key].append(value)
    else:
        temp_dict[key] = [value]

print(temp_dict)

from itertools import groupby


# final_dict = {}
# for key, group in groupby(mrg, lambda x: x[0]):
#     final_dict[key] = [g[1] for g in group]
#
#     # key_and_group = {key: list(g[1] for g in group)}
#     # for g in group:
#     #     print(type(g))
#     #     print(g[1])
#     # print()
# print(final_dict)

# from itertools import groupby
#
# data = [(1, "A"), (2, "B"), (1, "C"), (2, "D"), (3, "E")]
#
# result = {}
# for key, items in groupby(data, key=lambda x: x[0]):
#     result[key] = [item[1] for item in items]
#
# print(result)

# from itertools import groupby
#
# data = [("dog", "brown"), ("cat", "black"), ("dog", "white"),
#         ("bird", "yellow"), ("cat", "grey"), ("bird", "red")]
#
# result = {}
# for key, items in groupby(data, key=lambda x: x[0]):
#     result[key] = [item[1] for item in items]
#
# print(result)


from itertools import groupby

# data = [("dog", "brown"), ("cat", "black"), ("dog", "white"),
#         ("bird", "yellow"), ("cat", "grey"), ("bird", "red")]
#
# temp_dict = {}
# for key, value in data:
#     if key in temp_dict:
#         temp_dict[key].append(value)
#     else:
#         temp_dict[key] = [value]
#
# print(temp_dict)
# result = {key: values for key, values in temp_dict.items()}
#
# print(result)

# import time
# start_time = time.time()
# time.sleep(4)
# end_time = time.time()
# total = round((end_time - start_time)/60,2)
# print(total)
data_path = '/home/chris/Desktop/clip_feat_ViT-B_32_val_ic.pkl'

with open(data_path, 'rb') as f:
    all_data = pickle.load(f)

print(all_data.get('captions')[-1])

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_nod=8 --master_port=29500 multi_gpu_gptj.py
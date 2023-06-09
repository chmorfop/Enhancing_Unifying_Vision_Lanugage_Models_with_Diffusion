import pickle
import torch

with open('./data/textcaps/batch_0_diffgen_clipscore_clip_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    batch_0 = pickle.load(f)
with open('./data/textcaps/batch_1_diffgen_clipscore_clip_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    batch_1 = pickle.load(f)
with open('./data/textcaps/batch_2_diffgen_clipscore_clip_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    batch_2 = pickle.load(f)
with open('./data/textcaps/batch_3_diffgen_clipscore_clip_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    batch_3 = pickle.load(f)
with open('./data/textcaps/batch_4_diffgen_clipscore_clip_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    batch_4 = pickle.load(f)
with open('./data/textcaps/batch_5_diffgen_clipscore_clip_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    batch_5 = pickle.load(f)
with open('./data/textcaps/batch_6_diffgen_clipscore_clip_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    batch_6 = pickle.load(f)
with open('./data/textcaps/batch_7_diffgen_clipscore_clip_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    batch_7 = pickle.load(f)


batch_0_em = batch_0.get('clip_embedding')
batch_1_em = batch_1.get('clip_embedding')
batch_2_em = batch_2.get('clip_embedding')
batch_3_em = batch_3.get('clip_embedding')
batch_4_em = batch_4.get('clip_embedding')
batch_5_em = batch_5.get('clip_embedding')
batch_6_em = batch_6.get('clip_embedding')
batch_7_em = batch_7.get('clip_embedding')

final_em = torch.cat((batch_0_em, batch_1_em,batch_2_em,batch_3_em, batch_4_em,batch_5_em,batch_6_em,batch_7_em), dim=0)

batch_0_captions = batch_0.get('captions')
batch_1_captions = batch_1.get('captions')
batch_2_captions = batch_2.get('captions')
batch_3_captions = batch_3.get('captions')
batch_4_captions = batch_4.get('captions')
batch_5_captions = batch_5.get('captions')
batch_6_captions = batch_6.get('captions')
batch_7_captions = batch_7.get('captions')

final_captions = batch_0_captions + batch_1_captions + batch_2_captions + batch_3_captions + batch_4_captions + batch_5_captions + batch_6_captions + batch_7_captions

out_path = "./data/textcaps/diffgen_clipscore_feat_ViT-B_32_train_ic.pkl"
with open(out_path, 'wb') as f:
    pickle.dump({"clip_embedding": final_em, "captions": final_captions}, f)

print('Done!')
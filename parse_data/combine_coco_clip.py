import pickle
import torch




out_path = "./data/coco/combined_gen_clipscore_feat_train_ic.pkl"

with open('./data/coco/clip_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    old = pickle.load(f)

with open('./data/coco/diffgen_clipscore_feat_ViT-B_32_train_ic.pkl', 'rb') as f:
    new = pickle.load(f)

old_em = old.get('clip_embedding')
new_em = new.get('clip_embedding')

print(old_em.shape)
print(new_em.shape)
print('**' * 8)
print('**' * 8)

old_cap = old.get('captions')
new_cap = new.get('captions')

print(old_cap[-1])
print(new_cap[0])

for n in new_cap:
    old_cap.append({'caption': n.get('caption'),
                    'clip_embedding': n.get('clip_embedding') + 414113,
                    'image_id': n.get('image_id')})

print(old_cap[-1])

final_em = torch.cat((old_em, new_em), dim=0)
print(final_em.shape)

with open(out_path, 'wb') as f:
    pickle.dump({"clip_embedding": final_em, "captions": old_cap}, f)

print('Done!')

import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import skimage.io as io
import PIL.Image
import numpy as np
import clip
import time

CPU = torch.device("cpu")


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        temp_ans = self.answers[item]
        temp_q = self.questions[item]
        tokenized_answer = torch.tensor(self.tokenizer.encode(temp_ans), dtype=torch.int64)
        q_range = len(self.tokenizer.encode(temp_q))
        a_range = len(self.tokenizer.encode(temp_ans)) + 1
        rest_range = self.max_seq_len - q_range - a_range
        if rest_range >= 0:
            need_pred = q_range * [0] + a_range * [1] + rest_range * [0]
            need_pred_4gpt = q_range * [1] + a_range * [1] + rest_range * [0]
        elif rest_range < 0:
            # TODO
            # print('SOOS')
            need_pred = self.max_seq_len * [0]
            need_pred_4gpt = self.max_seq_len * [0]

        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        # A boolean tensor that is True where input is greater than or equal to other and False elsewhere
        # mask = tokens.ge(0)  # mask is zero where we out of sequence
        # tokens[~mask] = 0
        mask = torch.FloatTensor(need_pred)
        mask4gpt = torch.FloatTensor(need_pred_4gpt)

        omask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~omask] = 0

        # SOS
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        mask4gpt = torch.cat((torch.ones(self.prefix_length), mask4gpt), dim=0)  # adding prefix mask
        return tokens, mask, mask4gpt

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask, mask4gpt = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        # tokenized caption, mask attention , (prefix --> actual image)
        return tokens, mask, mask4gpt, prefix

    def __init__(self, data_path: str, prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        # image ids kai captions
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.answers = [caption['answer'] for caption in captions_raw]
        self.questions = [caption['question'] for caption in captions_raw]
        ##
        self.captions_tokens = []
        self.caption2embedding = []
        # self.temp_answers_tens = []
        eos = self.tokenizer.eos_token_id
        max_seq_len = 0
        max_ans_len = 0
        for i, caption in enumerate(captions_raw):
            # tokenize to caption
            self.captions_tokens.append(
                torch.tensor(self.tokenizer.encode(caption['question'] + ' ' + caption['answer']) + [eos],
                             dtype=torch.int64))
            # clip_embedding einai to sequential ID !!
            self.caption2embedding.append(caption["clip_embedding"])
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])

            temp = torch.tensor(self.tokenizer.encode(caption['answer']), dtype=torch.int64)
            max_ans_len = max(max_ans_len, temp.shape[0])

        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        self.max_ans_len = max_ans_len
        print('max_seq_len of whole tokens :  ' + str(self.max_seq_len))
        print('max_ans_len of answers :  ' + str(self.max_ans_len))


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        print('Initiate Transformer with 8 layers ***')
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        print('Initiate TransformerMapper *** ')
        self.clip_length = clip_length

        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        # edw erxontai ta tokens twn captions
        # ta prefix clip embedings
        # kai i maska

        # ta embeddings tou GPT gia to ekastwte token
        #  sinithws 1 x 30 x 768
        embedding_text = self.gpt.transformer.wte(tokens)
        # sinithws 1 x 10 x 768

        # kai prefix einai 1x512           ----- self.gpt_embedding_size 768
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        # kanoume concatenate ta prefix_projections & embedding_text
        # concat 1 x 40 x 768
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        #TODO ti attention mask vazw gia to VQA?? pws kataskeazete? se poio apo ta 2 kanei focus?
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        print()
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        print('Initiating the ClipCaptionModel *** ')
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


def train(dataset: ClipCocoDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    #device = torch.device('cuda:0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)

            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]

            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model



def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
    question = ''
):
    model.eval()
    # allo token index for . kai allo for EOS // tokenizer.eos_token_id
    stop_token_index = tokenizer.encode(stop_token)[0]
    eos_token_index = tokenizer.eos_token_id
    filter_value = -float("Inf")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                #generated = embed
                tok_q = torch.tensor(tokenizer.encode(question)).to(device)
                embedding_text = model.gpt.transformer.wte(tok_q).unsqueeze(0)
                generated = torch.cat((embed, embedding_text), dim=1)


            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                # if first time else concar
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if (stop_token_index == next_token.item() ) or eos_token_index == next_token.item() :
                    break

            output_text = tokenizer.decode(tokens.squeeze().cpu().numpy(), skip_special_tokens=True)

    return output_text

class Predictor():

    def __init__(self,weights_path):
        """Load the model into memory to make running multiple predictions efficient"""
        print('Initiating the Predictor')
        print('Utilizing weights path : ' + str(weights_path))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = 10
        mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}['transformer']
        model = ClipCaptionPrefix(self.prefix_length, clip_length = 10,
                                  prefix_size = 512, num_layers = 8, mapping_type = mapping_type)
        model.load_state_dict(torch.load(weights_path, map_location=CPU))
        model = model.eval()
        model = model.to(self.device)
        self.model = model

    def predict(self, image_path,question, use_beam_search):
        """Run a single prediction on the model"""
        image = io.imread(image_path)
        model = self.model
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(
                self.device, dtype=torch.float32
            )

            prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(model, self.tokenizer,question = question, embed=prefix_embed)


    def predict_fast(self, prefix,question, use_beam_search):
        """Run a single prediction on the model"""
        model = self.model
        prefix_embed = model.clip_project(prefix)
        if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(model, self.tokenizer,question = question, embed=prefix_embed)
    def generate_per_batch(self, batch_size, masky):
        tokens = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        stop_token_index = tokenizer.encode('.')[0]
        eos_token_index = tokenizer.eos_token_id
        max_length = 67
        temperature = 1.0
        seq_lengths = torch.ones(batch_size, device=device)
        is_stopped = torch.zeros(batch_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            for i in range(max_length):
                outputs = self.model.gpt(inputs_embeds=generated, attention_mask=masky)

                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                scores, next_tokens = logits.topk(1, -1)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                next_token_embed = self.model.gpt.transformer.wte(next_tokens)
                generated = torch.cat((generated, next_token_embed), dim=1)
                masky = torch.cat((masky, torch.ones((batch_size, 1), dtype=torch.float)), dim=1)

                seq_lengths[~is_stopped] += 1
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze() + \
                             next_tokens.eq(eos_token_index).squeeze()
                if is_stopped.all():
                    break

            output_list = tokens.cpu().numpy()
            output_texts = [
                tokenizer.decode(output[: int(length)], skip_special_tokens=True)
                for output, length in zip(output_list, seq_lengths)]
        return output_texts
    def custom_padding(self,q):
        max_len = 22
        padding = max_len - q.shape[0]
        if padding > 0:
            tokens = torch.cat((q, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = q[:max_len]
        q_range = q.shape[0]
        rest_range = max_len - q_range
        if rest_range >= 0:
            need_pred = q_range * [1] + rest_range * [0]

        omask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~omask] = 0
        mask = torch.FloatTensor(need_pred)
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens.unsqueeze(0), mask.unsqueeze(0)
    def predict_batch_images(self,images,questions):

        temperature = 1.0
        seq_lengths = torch.ones(12, device=self.device)
        is_stopped = torch.zeros(12, device=self.device, dtype=torch.bool)
        stop_token_index = self.tokenizer.encode('.')[0]
        eos_token_index = self.tokenizer.eos_token_id

        model = self.model
        gather_images = []
        for t in images:
            image = io.imread(t)
            pil_image = PIL.Image.fromarray(image)
            gather_images.append(self.preprocess(pil_image).unsqueeze(0).to(self.device))
        images = torch.cat(gather_images, dim=0)

        questions_list = []
        tokens_list = []
        mask_list = []
        eos = self.tokenizer.eos_token_id
        for q in questions:
            questions_list.append(torch.tensor(self.tokenizer.encode(q) + [eos]))
        for q in questions_list:
            tokens, mask = self.custom_padding(q)
            tokens_list.append(tokens)
            mask_list.append(mask)

        tokens = None
        temp_questions = torch.cat(tokens_list, dim=0)
        temp_qmasks = torch.cat(mask_list, dim=0)
        prefix = self.clip_model.encode_image(images).to(self.device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix)
        embedding_text = model.gpt.transformer.wte(temp_questions)
        generated = torch.cat((prefix_embed, embedding_text), dim=1)
        with torch.no_grad():
            for i in range(30):
                outputs = model.gpt(inputs_embeds=generated,attention_mask=temp_qmasks)
                logits = outputs.logits
                logits = logits[:, -1, :]
                logits = logits.softmax(-1).log()
                scores, next_tokens = logits.topk(1, -1)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                next_token_embed = model.gpt.transformer.wte(next_tokens)
                generated = torch.cat((generated, next_token_embed), dim=1)
                temp_qmasks = torch.cat((temp_qmasks, torch.ones((12, 1), dtype=torch.float)), dim=1)

                seq_lengths[~is_stopped] += 1
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze() \
                             + next_tokens.eq(eos_token_index).squeeze()
                if is_stopped.all():
                    break

            output_list = tokens.cpu().numpy()
            output_texts = [
                self.tokenizer.decode(output[: int(length)], skip_special_tokens=True)
                for output, length in zip(output_list, seq_lengths)]
        return output_texts



if __name__ == '__main__':

    gen = {}
    gts = {}
    full_gt_dict = {}
    uni_dict = {}
    start_time = time.time()

    # todo
    temp_path  = 'checkpoints/my_coco_vqa_model_bestmodel.pt'
    mypredictor = Predictor(weights_path=temp_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # todo
    val_dataset = ClipCocoDataset('/home/chris/PycharmProjects/CLIP_prefix_caption/data/coco/oscar_split_ViT-B_32_trainy_vqa_1024.pkl',
                                  10,normalize_prefix=False)
    qs = val_dataset.questions
    ans = val_dataset.answers
    im = val_dataset.prefixes
    imid = val_dataset.image_ids

    for i,val in tqdm(enumerate(qs),total=len(qs)):
        temp = im[i].to(device, dtype=torch.float32)
        output = mypredictor.predict_fast(prefix=temp.unsqueeze(0),
                                          question=qs[i],
                                          use_beam_search=False)
        gen[str(i)] = [output]
        gts[str(i)] = [ans[i]]
        full_gt_dict[str(i)] = {'image_id': imid[i],
                                'question': qs[i],
                                'answer': ans[i],
                                'predicted_answer': output
                                }

    with open("./full_gt_dict_vqa.json", "w") as outfile:
        json.dump(full_gt_dict, outfile)

    uni_dict['gen'] = gen
    uni_dict['gts'] = gts

    with open("./full_gt_dict_vqa.json", "w") as outfile:
        json.dump(full_gt_dict, outfile)

    with open("./uni_dict_vqa.json", "w") as outfile:
        json.dump(full_gt_dict, outfile)

    end_time = time.time()
    total = round((end_time - start_time) / 60, 2)
    print('*** The Validation is finished in {} minutes ***'.format(total))





import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import argparse
import os

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_model(model_name):
    if model_name == 'mpnet':
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    elif model_name == 'gte':
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        model = AutoModel.from_pretrained("thenlper/gte-large")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, tokenizer


def extract(model, tokenizer, model_name, path, extract_from, output_path):
    final_path = os.path.join(path, extract_from)
    output_name = extract_from.replace('/', '-')
    final_output_path = os.path.join(output_path, f'{output_name}.npy')
    all_features = []
    with torch.no_grad():
        with open('{}/seq.in'.format(final_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(final_path), 'r', encoding="utf-8") as f_label:
            for query, label in tqdm(zip(f_text, f_label), desc=f'Extract {extract_from}'):
                if model_name == 'mpnet':
                    batch_dict = tokenizer(query.strip(), padding=True, truncation=True, return_tensors='pt')
                    outputs = model(**batch_dict)
                    features = mean_pooling(outputs, batch_dict['attention_mask'])
                else:
                    batch_dict = tokenizer(query.strip(), max_length=512, padding=True, truncation=True,
                                           return_tensors='pt')
                    outputs = model(**batch_dict)
                    features = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                all_features.append(features)
        all_features = torch.cat(all_features, dim=0).contiguous().cpu().numpy()
        np.save(final_output_path, all_features)

def main(args):
    dataset = args.dataset
    model_name = args.model

    output_path = f'./features/{dataset}/{model_name}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model, tokenizer = get_model(model_name)

    if dataset == 'banking77':
        path = './data/BANKING77-OOS'
    elif dataset == 'clinc_banking':
        path = './data/CLINC-Single-Domain-OOS/banking'
    elif dataset == 'clinc_credit_cards':
        path = './data/CLINC-Single-Domain-OOS/credit_cards'
    else:
        raise ValueError('Invalid dataset')

    extract(model, tokenizer, model_name, path, 'train', output_path)   # Train normal data
    extract(model, tokenizer, model_name, path, 'test', output_path)    # Test normal data
    extract(model, tokenizer, model_name, path, 'id-oos/valid', output_path)   # Train OOS data
    extract(model, tokenizer, model_name, path, 'id-oos/test', output_path)   # Test OOS data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='banking77', choices=['banking77', 'clinc_banking', 'clinc_credit_cards'])
    parser.add_argument('--model', default='gte', choices=['gte', 'mpnet'])
    args = parser.parse_args()
    main(args)
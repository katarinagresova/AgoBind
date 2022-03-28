import torch
import random
#from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset
import pandas as pd
import numpy as np


def preprocess_datasets(dset, tokenizer, max_seq_len):
    '''
    Currently 
    AABBC with stride = 2 and kmer = 2 produces
    ['AA','BB'] and cuts off C 
    TODO is this desired? If not, set for cycle limit to len(text) only
    '''

    if(max_seq_len >=512):
        print('WARNING: some sequences are longer than 512, these are being trimmed')

    def coll(data):

        encoded_dset = [(label, tokenizer([[miRNA, gene]], max_length=max_seq_len, padding="max_length", truncation=True, return_tensors="pt"))
            for miRNA, gene, label in data[['miRNA', 'gene', 'label']].to_numpy()]
        encoded_samples = [
            {
                "input_ids": ids_dict['input_ids'][0],
                "attention_mask": ids_dict['attention_mask'][0],
                "token_type_ids": ids_dict['token_type_ids'][0],
                "labels": torch.tensor(label)
            }
            for label, ids_dict in encoded_dset
        ]

        return encoded_samples

    encoded_samples = coll(dset)

    return encoded_samples

def get_preprocessed_datasets(train_data, test_data, tokenizer, kmer_len, stride):

    train_dset = pd.read_csv(train_data, sep='\t')
    train_dset['miRNA'] = train_dset['miRNA'].apply(lambda x: ' '.join([x[i:i+kmer_len] for i in range(0, len(x)-kmer_len+1, stride)]))
    train_dset['gene'] = train_dset['gene'].apply(lambda x: ' '.join([x[i:i+kmer_len] for i in range(0, len(x)-kmer_len+1, stride)]))
    max_seq_len = max([train_dset.loc[i]['miRNA'].count(' ') + train_dset.loc[i]['gene'].count(' ') + 2 for i in range(len(train_dset))])
    max_seq_len = max_seq_len if max_seq_len < 512 else 512 
    encoded_samples = preprocess_datasets(train_dset, tokenizer, max_seq_len)
    random.shuffle(encoded_samples) #Shuffle the train set

    test_dset = pd.read_csv(test_data, sep='\t')
    encoded_samples_test = preprocess_datasets(test_dset, tokenizer, max_seq_len)

    return encoded_samples, encoded_samples_test



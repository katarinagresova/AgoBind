import torch
import random
#from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset
import pandas as pd


def preprocess_datasets(dset, tokenizer, kmer_len, stride, max_seq_len):
    '''
    Currently 
    AABBC with stride = 2 and kmer = 2 produces
    ['AA','BB'] and cuts off C 
    TODO is this desired? If not, set for cycle limit to len(text) only
    '''

    offset = kmer_len

    if(max_seq_len >=512):
        print('WARNING: some sequences are longer than 512, these are being trimmed')

    def coll(data):
        encoded_dset = [(label, tokenizer([text[i:i+offset] 
                        for i in range(0, len(text)-offset+1, stride)], max_length=max_seq_len, padding="max_length", is_split_into_words=True, truncation=True, verbose=True).input_ids)
                        for text, label in data]
        encoded_samples = [{"input_ids": torch.tensor(ids), "attention_mask": torch.tensor([1]*len(ids)), "labels": torch.tensor(label)} 
                    for label, ids in encoded_dset]

        return encoded_samples

    encoded_samples = coll(dset)

    return encoded_samples



def get_preprocessed_datasets(train_data, test_data, tokenizer, kmer_len, stride):

    train_dset = pd.read_csv(train_data, sep='\t')
    train_dset['seq'] = train_dset.apply(lambda x: x['miRNA'] + 'NNNN' + x['gene'], axis=1)
    max_seq_len = max([len(train_dset.loc[i]['seq']) for i in range(len(train_dset))])
    max_seq_len = max_seq_len if max_seq_len < 512 else 512 
    encoded_samples = preprocess_datasets(train_dset[['seq', 'label']].to_numpy(), tokenizer, kmer_len, stride, max_seq_len)
    random.shuffle(encoded_samples) #Shuffle the train set

    test_dset = pd.read_csv(test_data, sep='\t')
    test_dset['seq'] = test_dset.apply(lambda x: x['miRNA'] + 'NNNN' + x['gene'], axis=1)
    encoded_samples_test = preprocess_datasets(test_dset[['seq', 'label']].to_numpy(), tokenizer, kmer_len, stride, max_seq_len)

    return encoded_samples, encoded_samples_test
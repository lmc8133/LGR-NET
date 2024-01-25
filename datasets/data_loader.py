# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

#  https://daiwk.github.io/posts/nlp-bert-code-annotated-application.html

import os
import re
# import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
# import scipy.io as sio
import torch.utils.data as data
sys.path.append('.')

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.word_utils import Corpus
import pickle
import pdb

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)           #???
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class DatasetNotFoundError(Exception):
    pass

class TransVGDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')},
        'VG': {'splits': ('train', 'val', 'test')},
    }

    def __init__(self, data_root, split_root='data', dataset='referit', 
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, max_knowledge_len=256, lstm=False, 
                 bert_model='bert-base-uncased', use_knowledge = True, split_knowledge_query = True):
        self.images = []
        self.images_size=[]
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.knowledge_len = max_knowledge_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.return_idx = return_idx
        self.use_knowledge = use_knowledge
        self.split_knowledge_query = split_knowledge_query

        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.im_dir = osp.join(self.data_root, 'referit', 'images')
        elif  self.dataset == 'flickr':
            self.im_dir = osp.join(self.data_root, 'Flickr30k', 'flickr30k_images') # not download
        elif self.dataset == 'VG':
            self.im_dir = osp.join(self.data_root, '..' '..', '..', 'what-is-where-by-looking-main', 'data', 'visual_genome', 'VG_Images')
        elif self.dataset == 'SK':
            self.im_dir = osp.join(self.data_root, '..', '..', '..', 'datasets', 'sk-vg.v1', 'images')
        else:   ## refcoco, etc.
            self.im_dir = osp.join(self.data_root, '..', '..', '..', 'datasets', 'coco2014', 'train2014')

        # if self.dataset != 'VG' and not self.exists_dataset():
        #     # self.process_dataset()
        #     print('Please download index cache to data folder: \n \
        #         https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
        #     exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset == 'VG':
            if split == 'val':
                split = 'test'
            with open(self.im_dir + '/../VG_Annotations/data_splits.pickle', 'rb') as f:
                VG_splits = pickle.load(f)
            with open(self.im_dir + '/../VG_Annotations/region_descriptions.json', 'r', encoding = 'utf8') as f:
                desp = json.load(f)
            for itm in desp:
                if itm['id'] in VG_splits[split]:
                    img_file = str(itm['id']) + '.jpg'
                    for region in itm['regions']:
                        bbox = [region['x'], region['y'], region['width'], region['height']]
                        phrase = region['phrase']
                        self.images.append([img_file, bbox, phrase])
        else:
            if self.dataset != 'referit':
                splits = ['train', 'val'] if split == 'trainval' else [split]
            for split in splits:
                imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
                imgset_path = osp.join(dataset_path, imgset_file)
                self.images += torch.load(imgset_path)



    def exists_dataset(self):
        cwd=os.getcwd()
        path=osp.join(self.split_root,self.dataset)
        t=osp.exists(osp.join(self.split_root, "refer"))
        tt=osp.exists(path)
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr' or self.dataset == 'VG' or self.dataset == 'SK':
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")

        bbox = torch.tensor(bbox)
        bbox = bbox.float()
        return img, phrase, bbox

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        if self.dataset == 'SK' and self.use_knowledge and self.split_knowledge_query:
            phrase[0] = phrase[0].lower()   # query
            phrase[1] = phrase[1].lower()   # knowledge
        else:
            phrase = phrase.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        phrase = input_dict['text']
        img_mask = input_dict['mask']
        
        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id>0, dtype=int)
        else:
            ## encode phrase to bert input
            if self.dataset == 'SK' and self.use_knowledge and self.split_knowledge_query:
                query_examples = read_examples(phrase[0], idx)
                query_features = convert_examples_to_features(
                    examples = query_examples, seq_length=self.query_len, tokenizer=self.tokenizer)
                query_id = query_features[0].input_ids
                query_mask = query_features[0].input_mask

                knowledge_examples = read_examples(phrase[1], idx)
                knowledge_features = convert_examples_to_features(
                    examples = knowledge_examples, seq_length=self.knowledge_len, tokenizer=self.tokenizer)
                knowledge_id = knowledge_features[0].input_ids
                knowledge_mask = knowledge_features[0].input_mask
            else:
                examples = read_examples(phrase, idx)
                features = convert_examples_to_features(
                    examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
                word_id = features[0].input_ids
                word_mask = features[0].input_mask
        
        if self.testmode:   #always false
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int),\
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0]
        else:
            # print(img.shape)
            if self.dataset == 'SK' and self.use_knowledge and self.split_knowledge_query:
                return img, np.array(img_mask), np.array(query_id, dtype=int), np.array(query_mask, dtype=int), np.array(knowledge_id, dtype=int), np.array(knowledge_mask, dtype=int), np.array(bbox, dtype=np.float32)
            else:
                return img, np.array(img_mask), np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(bbox, dtype=np.float32)
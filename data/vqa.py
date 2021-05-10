"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VQA dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
import random

from .data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index


def _get_vqa_target(example, num_answers):
    target = torch.zeros(num_answers)
    labels = example['target']['labels']
    scores = example['target']['scores']
    if labels and scores:
        target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
    return target

def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


class VqaDataset(DetectFeatTxtTokDataset):
    def __init__(self, num_answers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_answers = num_answers
        self.text_only = False

    def set_text_only(self):
        self.text_only = True

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        # masked text input
        masked_input_ids, masked_txt_labels = self.create_mlm_io(example['input_ids'])

        target = _get_vqa_target(example, self.num_answers)

        if self.text_only:
            num_bb = 0
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, target, masked_input_ids, masked_txt_labels

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels

def get_vqa_collate(text_only=False):
    def vqa_collate_lambda(inputs):
        return vqa_collate(inputs, text_only=text_only)
    return vqa_collate_lambda

def vqa_collate(inputs, text_only=False):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets,
     masked_input_ids, masked_txt_labels) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    masked_input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=0)
    masked_txt_labels = pad_sequence(masked_txt_labels, batch_first=True, padding_value=-1)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)


    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    if text_only:
        num_bbs = [0 for f in img_feats]
        img_feat = None
        img_pos_feat = None

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'masked_input_ids': masked_input_ids,
             'masked_txt_labels': masked_txt_labels}
    return batch


class VqaEvalDataset(VqaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_mlm_inference = False

    def set_is_mlm_inference(self):
        self.is_mlm_inference = True


    def __getitem__(self, i):
        qid = self.ids[i]
        example = DetectFeatTxtTokDataset.__getitem__(self, i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        # masked text input
        masked_input_ids, masked_txt_labels = self.create_mlm_io(example['input_ids'], is_mlm_inference=self.is_mlm_inference)

        if 'target' in example:
            target = _get_vqa_target(example, self.num_answers)
        else:
            target = None

        if self.text_only:
            num_bb = 0
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return qid, input_ids, img_feat, img_pos_feat, attn_masks, target, masked_input_ids, masked_txt_labels

    def create_mlm_io(self, input_ids, is_mlm_inference=False):
        if is_mlm_inference:
            input_ids, txt_labels = self.create_txt_labels(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        else:
            input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels

    def create_txt_labels(self, tokens, vocab_range, mask):
        output_label = []

        for i, token in enumerate(tokens):
            if tokens[i] == mask:
                output_label.append(token)
            else:
                output_label.append(-1)

        return tokens, output_label

def get_vqa_eval_collate(text_only=False):
    def vqa_eval_collate_lambda(inputs):
        return vqa_eval_collate(inputs, text_only=text_only)
    return vqa_eval_collate_lambda

def vqa_eval_collate(inputs, text_only=False):
    (qids, input_ids, img_feats, img_pos_feats, attn_masks, targets,
     masked_input_ids, masked_txt_labels) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    masked_input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=0)
    masked_txt_labels = pad_sequence(masked_txt_labels, batch_first=True, padding_value=-1)

    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    if targets[0] is None:
        targets = None
    else:
        targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    
    if text_only:
        num_bbs = [0 for f in img_feats]
        img_feat = None
        img_pos_feat = None

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'qids': qids,
             'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'masked_input_ids': masked_input_ids,
             'masked_txt_labels': masked_txt_labels}
    return batch

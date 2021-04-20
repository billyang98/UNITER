import argparse
import json
import os
from os.path import exists
from time import time

import torch
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd
import numpy as np
from cytoolz import concat
from tqdm import tqdm

from data import (TokenBucketSampler, PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, VqaEvalDataset, vqa_eval_collate)
from model.vqa import UniterForVisualQuestionAnswering

from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct, parse_with_config
from utils.const import BUCKET_SIZE, IMG_DIM


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    hps_file = f'{opts.output_dir}/log/hps.json'
    model_opts = Struct(json.load(open(hps_file)))

    # train_examples = None
    ans2label_file = f'{opts.output_dir}/ckpt/ans2label.json'
    ans2label = json.load(open(ans2label_file))
    label2ans = {label: ans for ans, label in ans2label.items()}

    # load DBs and image dirs
    eval_img_db = DetectFeatLmdb(opts.img_db,
                                 model_opts.conf_th, model_opts.max_bb,
                                 model_opts.min_bb, model_opts.num_bb,
                                 False)
    eval_txt_db = TxtTokLmdb(opts.txt_db, -1)
    eval_dataset = VqaEvalDataset(len(ans2label), eval_txt_db, eval_img_db) 

    # Prepare model
    if exists(opts.checkpoint):
        ckpt_file = opts.checkpoint
    else:
        ckpt_file = f'{opts.output_dir}/ckpt/model_step_{opts.checkpoint}.pt'
    checkpoint = torch.load(ckpt_file)
    model = UniterForVisualQuestionAnswering.from_pretrained(
        f'{opts.output_dir}/log/model.json', checkpoint,
        img_dim=IMG_DIM, num_answer=len(ans2label))
    model.to(device)
    if opts.fp16:
        model = amp.initialize(model, enabled=True, opt_level='O2')

    sampler = TokenBucketSampler(eval_dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=opts.batch_size, droplast=False)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_sampler=sampler,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=vqa_eval_collate)
    eval_dataloader = PrefetchLoader(eval_dataloader)

    val_log, results, logits = evaluate(model, eval_dataloader, len(eval_dataset), label2ans,
                                        opts.save_logits)
    for k, v in val_log.items():
        print(f'{k} {v}')
    result_dir = f'{opts.output_dir}/results_test_{opts.test_name}'
    if not exists(result_dir) and rank == 0:
        os.makedirs(result_dir)

    all_results = list(concat(all_gather_list(results)))
    if opts.save_logits:
        all_logits = {}
        for id2logit in all_gather_list(logits):
            all_logits.update(id2logit)
    if hvd.rank() == 0:
        with open(f'{result_dir}/'
                  f'results_{opts.checkpoint}_all.json', 'w') as f:
            json.dump(all_results, f)
        if opts.save_logits:
            np.savez(f'{result_dir}/logits_{opts.checkpoint}_all.npz',
                     **all_logits)

@torch.no_grad()
def evaluate(model, eval_loader, eval_len, label2ans, save_logits=False, task='vqa'):
    LOGGER.info("start running evaluation {}...".format(task))
    model.eval()
    n_ex = 0
    tot_score = 0
    st = time()
    results = []
    logits = {}
    pbar = tqdm(total=eval_len)
    for i, batch in enumerate(eval_loader):
        qids = batch['qids']
        scores = model(batch, compute_loss=False, task=task)
        targets = batch['targets']
        tot_score += compute_score_with_logits(scores, targets).sum().item()
        answers = [label2ans[i]
                   for i in scores.max(dim=-1, keepdim=False
                                       )[1].cpu().tolist()]
        for qid, answer in zip(qids, answers):
            results.append({'answer': answer, 'question_id': qid})
        if save_logits:
            scores = scores.cpu()
            for i, qid in enumerate(qids):
                logits[qid] = scores[i].half().numpy()
        n_ex += len(qids)
        pbar.update(len(qids))
        # TODO: dont commit, for testing only
        #if i > 4:
        #    break
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    acc = tot_score / n_ex
    tot_time = time()-st
    val_log = {'valid/ex_per_s': n_ex/tot_time,
                'valid/acc': acc}
    model.train()
    LOGGER.info(f"evaluation finished in {int(tot_time)} seconds "
                f"at {int(n_ex/tot_time)} examples per second")
    return val_log, results, logits

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]  # argmax
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    main(args)

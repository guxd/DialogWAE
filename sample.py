"""
Copyright 2018 NAVER Corp.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import json

import os, sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import indexes2sent, gVar, gData
import models, experiments, data, configs
from models import DialogWAE, DialogWAE_GMP
from experiments import Metrics

PAD_token = 0

def evaluate(model, metrics, test_loader, vocab, ivocab, f_eval, repeat):
    
    recall_bleus, prec_bleus, bows_extrema, bows_avg, bows_greedy, intra_dist1s, intra_dist2s, avg_lens, inter_dist1s, inter_dist2s\
        = [], [], [], [], [], [], [], [], [], []
    local_t = 0
    while True:
        batch = test_loader.next_batch()
        if batch is None:
            break
        local_t += 1 
        context, context_lens, utt_lens, floors,_,_,_,response,res_lens,_ = batch   
        context, utt_lens = context[:,:,1:], utt_lens-1 # remove the sos token in the context and reduce the context length
        f_eval.write("Batch %d \n" % (local_t))# print the context
        start = np.maximum(0, context_lens[0]-5)
        for t_id in range(start, context.shape[1], 1):
            context_str = indexes2sent(context[0, t_id], vocab, vocab["</s>"], PAD_token)
            f_eval.write("Context %d-%d: %s\n" % (t_id, floors[0, t_id], context_str))
        # print the true outputs    
        ref_str, _ =indexes2sent(response[0], vocab, vocab["</s>"], vocab["<s>"])
        ref_tokens = ref_str.split(' ')
        f_eval.write("Target >> %s\n" % (ref_str.replace(" ' ", "'")))
        
        context, context_lens, utt_lens, floors = gVar(context), gVar(context_lens), gVar(utt_lens), gData(floors)
        sample_words, sample_lens = model.sample(context, context_lens, utt_lens, floors, repeat, vocab["<s>"], vocab["</s>"])
        # nparray: [repeat x seq_len]
        pred_sents, _ = indexes2sent(sample_words, vocab, vocab["</s>"], PAD_token)
        pred_tokens = [sent.split(' ') for sent in pred_sents]
        for r_id, pred_sent in enumerate(pred_sents):
            f_eval.write("Sample %d >> %s\n" % (r_id, pred_sent.replace(" ' ", "'")))
        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)
        
        bow_extrema, bow_avg, bow_greedy = metrics.sim_bow(sample_words, sample_lens, response[:,1:], res_lens-2)
        bows_extrema.append(bow_extrema)
        bows_avg.append(bow_avg)
        bows_greedy.append(bow_greedy)
        
        intra_dist1, intra_dist2, inter_dist1, inter_dist2 = metrics.div_distinct(sample_words, sample_lens)
        intra_dist1s.append(intra_dist1)
        intra_dist2s.append(intra_dist2)
        avg_lens.append(np.mean(sample_lens))
        inter_dist1s.append(inter_dist1)
        inter_dist2s.append(inter_dist2)
                
        f_eval.write("\n")

    recall_bleu = float(np.mean(recall_bleus))
    prec_bleu = float(np.mean(prec_bleus))
    f1 = 2*(prec_bleu*recall_bleu) / (prec_bleu+recall_bleu+10e-12)
    bow_extrema = float(np.mean(bows_extrema))
    bow_avg = float(np.mean(bows_avg))
    bow_greedy=float(np.mean(bows_greedy))
    intra_dist1=float(np.mean(intra_dist1s))
    intra_dist2=float(np.mean(intra_dist2s))
    avg_len=float(np.mean(avg_lens))
    inter_dist1=float(np.mean(inter_dist1s))
    inter_dist2=float(np.mean(inter_dist2s))
    report = "Avg recall BLEU %f, avg precision BLEU %f, F1 %f, bow_extrema %f, bow_avg %f, bow_greedy %f,\
    intra_dist1 %f, intra_dist2 %f, avg_len %f, inter_dist1 %f, inter_dist2 %f (only 1 ref, not final results)" \
    % (recall_bleu, prec_bleu, f1, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len, inter_dist1, inter_dist2)
    print(report)
    f_eval.write(report + "\n")
    print("Done testing")
    
    return recall_bleu, prec_bleu, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len, inter_dist1, inter_dist2

def main(args):
    # Load the models
    conf = getattr(configs, 'config_'+args.model)()
    model=torch.load(f='./output/{}/{}/{}/models/model_epo{}.pckl'.format(args.model, args.expname, args.dataset, args.reload_from))
    model.eval()
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")
    
    
    # Load data
    data_path=args.data_path+args.dataset+'/'
    glove_path = args.data_path+'glove.twitter.27B.200d.txt'
    corpus = getattr(data, args.dataset+'Corpus')(data_path, wordvec_path=glove_path, wordvec_dim=conf['emb_size'])
    dials, metas = corpus.get_dialogs(), corpus.get_metas()
    test_dial, test_meta = dials.get("test"), metas.get("test")
    # convert to numeric input outputs that fits into TF models
    test_loader = getattr(data, args.dataset+'DataLoader')("Test", test_dial, test_meta, conf['maxlen'])
    test_loader.epoch_init(1, conf['diaglen'], 1, shuffle=False)  
    ivocab = corpus.vocab
    vocab = corpus.ivocab
    
    metrics=Metrics(corpus.word2vec)
    
    f_eval = open("./output/{}/{}/{}/results.txt".format(args.model, args.expname, args.dataset), "w")
    repeat = args.n_samples
    
    evaluate(model, metrics, test_loader, vocab, ivocab, f_eval, repeat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DialogGAN for Eval')
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='SWDA', help='name of dataset, SWDA or DailyDial')
    parser.add_argument('--model', type=str, default='DialogWAE', help='model name')
    parser.add_argument('--expname', type=str, default='basic', help='experiment name, disinguishing different parameter settings')
    parser.add_argument('--reload_from', type=int, default=40, help='directory to load models from, SWDA 8, 40, DailyDial 6, 40')
    
    parser.add_argument('--n_samples', type=int, default=10, help='Number of responses to sampling')
    parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    print(vars(args))
    main(args)

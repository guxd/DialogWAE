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

import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity as cosine
from collections import Counter

class Metrics:
    """
    """
    def __init__(self, word2vec):
        """
        :param word2vec - a numpy array of word2vec with shape [vocab_size x emb_size]
        """
        super(Metrics, self).__init__()
        self.word2vec = word2vec
        
    def embedding(self, seqs): 
        """
        A numpy version of embedding
        :param seqs - ndarray [batch_sz x seqlen]
        """
        batch_size, seqlen = seqs.shape
        seqs=np.reshape(seqs, (-1)) # convert to 1-d indexes [(batch_sz*seqlen)]
        embs = self.word2vec[seqs] # lookup [(batch_sz*seqlen) x emb_sz]
        embs = np.reshape(embs, (batch_size, seqlen, -1)) # recover the shape [batch_sz x seqlen x emb_sz]
        return embs
    
    def extrema(self, embs, lens): # embs: [batch_size x seq_len x emb_size]  lens: [batch_size]
        """
        computes the value of every single dimension in the word vectors which has the greatest
        difference from zero.
        :param seq: sequence
        :param seqlen: length of sequence
        """
        # Find minimum and maximum value for every dimension in predictions
        batch_size, seq_len, emb_size = embs.shape
        max_mask = np.zeros((batch_size, seq_len, emb_size), dtype=np.int)
        for i,length in enumerate(lens):
            max_mask[i,:length,:]=1
        min_mask = 1-max_mask
        seq_max = (embs*max_mask).max(1) # [batch_sz x emb_sz]
        seq_min = (embs+min_mask).min(1)
        # Find the maximum absolute value in min and max data
        comp_mask = seq_max >= np.abs(seq_min)# [batch_sz x emb_sz]
        # Add vectors for finding final sequence representation for predictions
        extrema_emb = seq_max* comp_mask + seq_min* np.logical_not(comp_mask)
        return extrema_emb
    
    def mean(self, embs, lens):
        batch_size, seq_len, emb_size=embs.shape
        mask = np.zeros((batch_size, seq_len, emb_size), dtype=np.int)
        for i,length in enumerate(lens):
            mask[i,:length,:]=1
        return (embs*mask).sum(1)/(mask.sum(1)+1e-8)

    def sim_bleu(self, hyps, ref):
        """
        :param ref - a list of tokens of the reference
        :param hyps - a list of tokens of the hypothesis
    
        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            try:
                scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                        weights=[1./3, 1./3, 1./3]))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)


    def sim_bow(self, pred, pred_lens, ref, ref_lens):
        """
        :param pred - ndarray [batch_size x seqlen]
        :param pred_lens - list of integers
        :param ref - ndarray [batch_size x seqlen]
        """
        # look up word embeddings for prediction and reference
        emb_pred = self.embedding(pred) # [batch_sz x seqlen1 x emb_sz]
        emb_ref = self.embedding(ref) # [batch_sz x seqlen2 x emb_sz]
        
        ext_emb_pred=self.extrema(emb_pred, pred_lens)
        ext_emb_ref=self.extrema(emb_ref, ref_lens)
        bow_extrema=cosine(ext_emb_pred, ext_emb_ref) # [batch_sz_pred x batch_sz_ref]
        
        avg_emb_pred = self.mean(emb_pred, pred_lens) # Calculate mean over seq
        avg_emb_ref = self.mean(emb_ref, ref_lens) 
        bow_avg = cosine(avg_emb_pred, avg_emb_ref) # [batch_sz_pred x batch_sz_ref]

        
        batch_pred, seqlen_pred, emb_size=emb_pred.shape
        batch_ref, seqlen_ref, emb_size=emb_ref.shape
        cos_sim = cosine(emb_pred.reshape((-1, emb_size)), emb_ref.reshape((-1, emb_size))) # [(batch_sz*seqlen1)x(batch_sz*seqlen2)]
        cos_sim = cos_sim.reshape((batch_pred, seqlen_pred, batch_ref, seqlen_ref))
        # Find words with max cosine similarity
        max12 = cos_sim.max(1).mean(2) # max over seqlen_pred
        max21 = cos_sim.max(3).mean(1) # max over seqlen_ref
        bow_greedy=(max12+max21)/2 # [batch_pred x batch_ref(1)]
        return np.max(bow_extrema), np.max(bow_avg), np.max(bow_greedy)
    
    def div_distinct(self, seqs, seq_lens):
        """
        distinct-1 distinct-2 metrics for diversity measure proposed 
        by Li et al. "A Diversity-Promoting Objective Function for Neural Conversation Models"
        we counted numbers of distinct unigrams and bigrams in the generated responses 
        and divide the numbers by total number of unigrams and bigrams. 
        The two metrics measure how informative and diverse the generated responses are. 
        High numbers and high ratios mean that there is much content in the generated responses, 
        and high numbers further indicate that the generated responses are long
        """
        batch_size = seqs.shape[0]
        intra_dist1, intra_dist2=np.zeros(batch_size), np.zeros(batch_size)
        
        n_unigrams, n_bigrams, n_unigrams_total , n_bigrams_total = 0. ,0., 0., 0.
        unigrams_all, bigrams_all = Counter(), Counter()
        for b in range(batch_size):
            unigrams= Counter([tuple(seqs[b,i:i+1]) for i in range(seq_lens[b])])
            bigrams = Counter([tuple(seqs[b,i:i+2]) for i in range(seq_lens[b]-1)])
            intra_dist1[b]=(len(unigrams.items())+1e-12)/(seq_lens[b]+1e-5)
            intra_dist2[b]=(len(bigrams.items())+1e-12)/(max(0, seq_lens[b]-1)+1e-5)
            
            unigrams_all.update([tuple(seqs[b,i:i+1]) for i in range(seq_lens[b])])
            bigrams_all.update([tuple(seqs[b,i:i+2]) for i in range(seq_lens[b]-1)])
            n_unigrams_total += seq_lens[b]
            n_bigrams_total += max(0, seq_lens[b]-1)
        inter_dist1 = (len(unigrams_all.items())+1e-12)/(n_unigrams_total+1e-5)
        inter_dist2 = (len(bigrams_all.items())+1e-12)/(n_bigrams_total+1e-5)
        return intra_dist1, intra_dist2, inter_dist1, inter_dist2

    


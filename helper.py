import time
import math
import numpy as np

def asHHMMSS(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m /60)
    m -= h *60
    return '%d:%d:%d'% (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s<%s'%(asHHMMSS(s), asHHMMSS(rs))

#######################################################################
def sent2indexes(sentence, vocab):
    def convert_sent(sent, vocab):
        return np.array([vocab[word] for word in sent.split(' ')])
    if type(sentence) is list:
        indexes=[convert_sent(sent, vocab) for sent in sentence]
        sent_lens = [len(idxes) for idxes in indexes]
        max_len = max(sent_lens)
        inds = np.zeros((len(sentence), max_len), dtype=np.int)
        for i, idxes in enumerate(indexes):
            inds[i,:len(idxes)]=indexes[i]
        return inds
    else:
        return convert_sent(sentence, vocab)

def indexes2sent(indexes, vocab, eos_tok, ignore_tok=0): 
    '''indexes: numpy array'''
    def revert_sent(indexes, ivocab, eos_tok, ignore_tok=0):
        toks=[]
        length=0
        indexes=filter(lambda i: i!=ignore_tok, indexes)
        for idx in indexes:
            toks.append(ivocab[idx])
            length+=1
            if idx == eos_tok:
                break
        return ' '.join(toks), length
    
    ivocab = {v: k for k, v in vocab.items()}
    if indexes.ndim==1:# one sentence
        return revert_sent(indexes, ivocab, eos_tok, ignore_tok)
    else:# dim>1
        sentences=[] # a batch of sentences
        lens=[]
        for inds in indexes:
            sentence, length = revert_sent(inds, ivocab, eos_tok, ignore_tok)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens
import torch
from torch.nn import functional as F

use_cuda = torch.cuda.is_available()

def gData(data):
    tensor=data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if use_cuda:
        tensor=tensor.cuda()
    return tensor
def gVar(data):
    return gData(data)


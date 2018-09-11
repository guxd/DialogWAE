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
import time
from datetime import datetime
import numpy as np
import random
import json
import logging
import torch
import os, sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import timeSince, sent2indexes, indexes2sent, gData, gVar
import models, experiments, configs, data
from experiments import Metrics
from sample import evaluate

from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package

parser = argparse.ArgumentParser(description='DialogWAE Pytorch')
# Path Arguments
parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='SWDA', help='name of dataset. SWDA or DailyDial')
parser.add_argument('--model', type=str, default='DialogWAE_GMP', help='model name')
parser.add_argument('--expname', type=str, default='basic', help='experiment name, for disinguishing different parameter settings')
parser.add_argument('--visual', action='store_true', default=False, help='visualize training status in tensorboard')
parser.add_argument('--reload_from', type=int, default=-1, help='reload from a trained ephoch')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true', help='sample when decoding for generation')
parser.add_argument('--log_every', type=int, default=100, help='interval to log training results')
parser.add_argument('--valid_every', type=int, default=400, help='interval to validation')
parser.add_argument('--eval_every', type=int, default=2, help='interval to evaluate on the validation set')
parser.add_argument('--seed', type=int, default=1111, help='random seed')

args = parser.parse_args()
print(vars(args))

# make output directory if it doesn't already exist
if not os.path.isdir('./output'):
    os.makedirs('./output')
if not os.path.isdir('./output/{}'.format(args.model)):
    os.makedirs('./output/{}'.format(args.model))
if not os.path.isdir('./output/{}/{}'.format(args.model, args.expname)):
    os.makedirs('./output/{}/{}'.format(args.model, args.expname))
if not os.path.isdir('./output/{}/{}/{}'.format(args.model, args.expname, args.dataset)):
    os.makedirs('./output/{}/{}/{}'.format(args.model, args.expname, args.dataset))
if not os.path.isdir('./output/{}/{}/{}/models'.format(args.model, args.expname, args.dataset)):
    os.makedirs('./output/{}/{}/{}/models'.format(args.model, args.expname, args.dataset))
if not os.path.isdir('./output/{}/{}/{}/tmp_results'.format(args.model, args.expname, args.dataset)):
    os.makedirs('./output/{}/{}/{}/tmp_results'.format(args.model, args.expname, args.dataset))

# save arguments
json.dump(vars(args), open('./output/{}/{}/{}/args.json'.format(args.model, args.expname, args.dataset), 'w'))

# LOG #
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")#,format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
fh = logging.FileHandler("./output/{}/{}/{}/logs.txt".format(args.model, args.expname, args.dataset))
                                  # create file handler which logs even debug messages
logger.addHandler(fh)# add the handlers to the logger

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu_id) # set gpu device
    torch.cuda.manual_seed(args.seed)

def save_model(model, epoch):
    print("Saving models")
    torch.save(f='./output/{}/{}/{}/models/model_epo{}.pckl'.format(args.model, args.expname, args.dataset, epoch),obj=model)
def load_model(epoch):
    print("Loading models")
    model = torch.load(f='./output/{}/{}/{}/models/model_epo{}.pckl'.format(args.model, args.expname, args.dataset, epoch))
    return model

config = getattr(configs, 'config_'+args.model)()

###############################################################################
# Load data
###############################################################################
data_path=args.data_path+args.dataset+'/'
#
corpus = getattr(data, args.dataset+'Corpus')(data_path, wordvec_path=args.data_path+'glove.twitter.27B.200d.txt', wordvec_dim=config['emb_size'])
dials = corpus.get_dialogs()
metas = corpus.get_metas()
train_dial, valid_dial, test_dial = dials.get("train"), dials.get("valid"), dials.get("test")
train_meta, valid_meta, test_meta = metas.get("train"), metas.get("valid"), metas.get("test")
train_loader = getattr(data, args.dataset+'DataLoader')("Train", train_dial, train_meta, config['maxlen'])
valid_loader = getattr(data, args.dataset+'DataLoader')("Valid", valid_dial, valid_meta, config['maxlen'])
test_loader = getattr(data, args.dataset+'DataLoader')("Test", test_dial, test_meta, config['maxlen'])

vocab = corpus.ivocab
ivocab = corpus.vocab
n_tokens = len(ivocab)

metrics=Metrics(corpus.word2vec)

print("Loaded data!")

###############################################################################
# Define the models
###############################################################################

model = getattr(models, args.model)(config, n_tokens) if args.reload_from<0 else load_model(args.reload_from)
if use_cuda:
    model=model.cuda()
    
if corpus.word2vec is not None and args.reload_from<0:
    print("Loaded word2vec")
    model.embedder.weight.data.copy_(torch.from_numpy(corpus.word2vec))
    model.embedder.weight.data[0].fill_(0)

tb_writer = SummaryWriter("./output/{}/{}/{}/logs/".format(args.model, args.expname, args.dataset)\
                          +datetime.now().strftime('%Y%m%d%H%M')) if args.visual else None

logger.info("Training...")
itr_global=1
start_epoch=1 if args.reload_from==-1 else args.reload_from+1
for epoch in range(start_epoch, config['epochs']+1):

    epoch_start_time = time.time()
    itr_start_time = time.time()
    
    # shuffle (re-define) data between epochs   
    train_loader.epoch_init(config['batch_size'], config['diaglen'], 1, shuffle=True)
 
    n_iters=train_loader.num_batch/max(1, config['n_iters_d'])
    
    itr = 1
    while True:# loop through all batches in training data
        model.train()
        loss_records=[]
        batch = train_loader.next_batch()
        if batch is None: # end of epoch
            break
        context, context_lens, utt_lens, floors,_,_,_,response,res_lens,_ = batch
        context, utt_lens = context[:,:,1:], utt_lens-1 # remove the sos token in the context and reduce the context length
        context, context_lens, utt_lens, floors, response, res_lens\
                = gVar(context), gVar(context_lens), gVar(utt_lens), gData(floors), gVar(response), gVar(res_lens)
            
        loss_AE = model.train_AE(context, context_lens, utt_lens, floors, response, res_lens)
        loss_records.extend(loss_AE)

        loss_G = model.train_G(context, context_lens, utt_lens, floors, response, res_lens)
        loss_records.extend(loss_G)
        
        for i in range(config['n_iters_d']):# train discriminator/critic
            loss_D = model.train_D(context, context_lens, utt_lens, floors, response, res_lens)  
            if i==0:
                loss_records.extend(loss_D)
            if i==config['n_iters_d']-1:
                break
            batch = train_loader.next_batch()
            if batch is None: # end of epoch
                break
            context, context_lens, utt_lens, floors,_,_,_,response,res_lens,_ = batch
            context, utt_lens = context[:,:,1:], utt_lens-1 # remove the sos token in the context and reduce the context length
            context, context_lens, utt_lens, floors, response, res_lens\
                = gVar(context), gVar(context_lens), gVar(utt_lens), gData(floors), gVar(response), gVar(res_lens)                      
                
                               
        if itr % args.log_every == 0:
            elapsed = time.time() - itr_start_time
            log = '%s-%s|%s@gpu%d epo:[%d/%d] iter:[%d/%d] step_time:%ds elapsed:%s \n                      '\
            %(args.model, args.expname, args.dataset, args.gpu_id, epoch, config['epochs'],
                     itr, n_iters, elapsed, timeSince(epoch_start_time,itr/n_iters))
            for loss_name, loss_value in loss_records:
                log=log+loss_name+':%.4f '%(loss_value)
                if args.visual:
                    tb_writer.add_scalar(loss_name, loss_value, itr_global)
            logger.info(log)
                
            itr_start_time = time.time()   
            
        if itr % args.valid_every == 0:
            valid_loader.epoch_init(config['batch_size'], config['diaglen'], 1, shuffle=False)
            model.eval()
            loss_records={}
            
            while True:
                batch = valid_loader.next_batch()
                if batch is None: # end of epoch
                    break
                context, context_lens, utt_lens, floors,_,_,_,response,res_lens,_ = batch
                context, utt_lens = context[:,:,1:], utt_lens-1 # remove the sos token in the context and reduce the context length
                context, context_lens, utt_lens, floors, response, res_lens\
                        = gVar(context), gVar(context_lens), gVar(utt_lens), gData(floors), gVar(response), gVar(res_lens)
                valid_loss = model.valid(context, context_lens, utt_lens, floors, response, res_lens)    
                for loss_name, loss_value in valid_loss:
                    v=loss_records.get(loss_name, [])
                    v.append(loss_value)
                    loss_records[loss_name]=v
                
            log = 'Validation '
            for loss_name, loss_values in loss_records.items():
                log = log + loss_name + ':%.4f  '%(np.mean(loss_values))
                if args.visual:
                    tb_writer.add_scalar(loss_name, np.mean(loss_values), itr_global)                 
            logger.info(log)    
            
        itr += 1
        itr_global+=1
        
        
    if epoch % args.eval_every == 0:  # evaluate the model in the validation set
        model.eval()               
        valid_loader.epoch_init(1, config['diaglen'], 1, shuffle=False) 
        
        f_eval = open("./output/{}/{}/{}/tmp_results/epoch{}.txt".format(args.model, args.expname, args.dataset, epoch), "w")
        repeat = 10
        
        recall_bleu, prec_bleu, bow_extrema, bow_avg, bow_greedy, intra_dist1, intra_dist2, avg_len, inter_dist1, inter_dist2\
             =evaluate(model, metrics, valid_loader, vocab, ivocab, f_eval, repeat)
                         
        if args.visual:
            tb_writer.add_scalar('recall_bleu', recall_bleu, epoch)
            tb_writer.add_scalar('prec_bleu', prec_bleu, epoch)
            tb_writer.add_scalar('bow_extrema', bow_extrema, epoch)
            tb_writer.add_scalar('bow_avg', bow_avg, epoch) 
            tb_writer.add_scalar('bow_greedy', bow_greedy, epoch) 
            tb_writer.add_scalar('intra_dist1', intra_dist1, epoch) 
            tb_writer.add_scalar('intra_dist2', intra_dist2, epoch) 
            tb_writer.add_scalar('inter_dist1', inter_dist1, epoch) 
            tb_writer.add_scalar('inter_dist2', inter_dist2, epoch) 

    # end of epoch ----------------------------
    model.adjust_lr()
    save_model(model, epoch) # save model after each epoch

    


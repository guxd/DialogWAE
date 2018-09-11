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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import os
import numpy as np
import random
import sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import gVar, gData
from modules import Encoder, ContextEncoder, Variation, Decoder              
    
one = gData(torch.FloatTensor([1]))
minus_one = one * -1    

class DialogWAE(nn.Module):
    def __init__(self, config, vocab_size, PAD_token=0):
        super(DialogWAE, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen=config['maxlen']
        self.clip = config['clip']
        self.lambda_gp = config['lambda_gp']
        self.temp=config['temp']
        
        self.embedder= nn.Embedding(vocab_size, config['emb_size'], padding_idx=PAD_token)
        self.utt_encoder = Encoder(self.embedder, config['emb_size'], config['n_hidden'], 
                                   True, config['n_layers'], config['noise_radius']) 
        self.context_encoder = ContextEncoder(self.utt_encoder, config['n_hidden']*2+2, config['n_hidden'], 1, config['noise_radius']) 
        self.prior_net = Variation(config['n_hidden'], config['z_size']) # p(e|c)
        self.post_net = Variation(config['n_hidden']*3, config['z_size']) # q(e|c,x)
        
        self.post_generator = nn.Sequential( 
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size'])
        )
        self.post_generator.apply(self.init_weights)
                                                                              
        self.prior_generator = nn.Sequential( 
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size']),
            nn.BatchNorm1d(config['z_size'], eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(config['z_size'], config['z_size'])
        ) 
        self.prior_generator.apply(self.init_weights)
                                                                                             
        self.decoder = Decoder(self.embedder, config['emb_size'], config['n_hidden']+config['z_size'], 
                               vocab_size, n_layers=1) 
        
        self.discriminator = nn.Sequential(  
            nn.Linear(config['n_hidden']+config['z_size'], config['n_hidden']*2),
            nn.BatchNorm1d(config['n_hidden']*2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config['n_hidden']*2, config['n_hidden']*2),
            nn.BatchNorm1d(config['n_hidden']*2, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(config['n_hidden']*2, 1),
        )
        self.discriminator.apply(self.init_weights)
        
           
        self.optimizer_AE = optim.SGD(list(self.context_encoder.parameters())
                                      +list(self.post_net.parameters())
                                      +list(self.post_generator.parameters())
                                      +list(self.decoder.parameters()),lr=config['lr_ae'])
        self.optimizer_G = optim.RMSprop(list(self.post_net.parameters())
                                      +list(self.post_generator.parameters())
                                      +list(self.prior_net.parameters())
                                      +list(self.prior_generator.parameters()), lr=config['lr_gan_g'])
        self.optimizer_D = optim.RMSprop(self.discriminator.parameters(), lr=config['lr_gan_d'])
        
        self.lr_scheduler_AE = optim.lr_scheduler.StepLR(self.optimizer_AE, step_size = 10, gamma=0.6)
        
        self.criterion_ce = nn.CrossEntropyLoss()
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.02, 0.02)
            m.bias.data.fill_(0)
            
    def sample_code_post(self, x, c):
        e, _, _ = self.post_net(torch.cat((x, c),1))
        z = self.post_generator(e)
        return z
   
    def sample_code_prior(self, c):
        e, _, _ = self.prior_net(c)
        z = self.prior_generator(e)
        return z    
    
    def train_AE(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.train()
        self.decoder.train()
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)      
        z = self.sample_code_post(x, c)
        output = self.decoder(torch.cat((z, c),1), None, response[:,:-1], (res_lens-1))  
        flattened_output = output.view(-1, self.vocab_size) 
        
        dec_target = response[:,1:].contiguous().view(-1)
        mask = dec_target.gt(0) # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask) # 
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)# [(batch_sz*seq_len) x n_tokens]
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        
        self.optimizer_AE.zero_grad()
        loss = self.criterion_ce(masked_output/self.temp, masked_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(self.context_encoder.parameters())+list(self.decoder.parameters()), self.clip)
        self.optimizer_AE.step()

        return [('train_loss_AE', loss.item())]        
    
    def train_G(self, context, context_lens, utt_lens, floors, response, res_lens): 
        self.context_encoder.eval()
        self.optimizer_G.zero_grad()
        
        for p in self.discriminator.parameters():
            p.requires_grad = False  
        
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        # -----------------posterior samples ---------------------------
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)
        z_post= self.sample_code_post(x.detach(), c.detach())
        errG_post = torch.mean(self.discriminator(torch.cat((z_post, c.detach()),1) ))
        errG_post.backward(minus_one) 
    
        # ----------------- prior samples ---------------------------
        prior_z = self.sample_code_prior(c.detach()) 
        errG_prior = torch.mean(self.discriminator(torch.cat((prior_z, c.detach()),1)))
        errG_prior.backward(one) 
    
        self.optimizer_G.step()
        
        for p in self.discriminator.parameters():
            p.requires_grad = True  
        
        costG = errG_prior - errG_post
        return [('train_loss_G', costG.item())]
    
    def train_D(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.eval()
        self.discriminator.train()
        
        self.optimizer_D.zero_grad()
        
        batch_size=context.size(0)

        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)
        post_z = self.sample_code_post(x, c)
        errD_post = torch.mean(self.discriminator(torch.cat((post_z.detach(), c.detach()),1)))
        errD_post.backward(one)
 
        prior_z = self.sample_code_prior(c) 
        errD_prior = torch.mean(self.discriminator(torch.cat((prior_z.detach(), c.detach()),1)))
        errD_prior.backward(minus_one) 
    
        alpha = gData(torch.rand(batch_size, 1))
        alpha = alpha.expand(prior_z.size())
        interpolates = alpha * prior_z.data + ((1 - alpha) * post_z.data)
        interpolates = Variable(interpolates, requires_grad=True)
        d_input=torch.cat((interpolates, c.detach()),1)
        disc_interpolates = torch.mean(self.discriminator(d_input))
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                               grad_outputs=gData(torch.ones(disc_interpolates.size())),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.contiguous().view(gradients.size(0),-1).norm(2,dim=1)-1)**2).mean()*self.lambda_gp
        gradient_penalty.backward()
    
        self.optimizer_D.step()
        costD = -(errD_prior - errD_post) + gradient_penalty
        return [('train_loss_D', costD.item())]   
    
    def valid(self, context, context_lens, utt_lens, floors, response, res_lens):
        self.context_encoder.eval()      
        self.discriminator.eval()
        self.decoder.eval()
        
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        x,_ = self.utt_encoder(response[:,1:], res_lens-1)
        post_z = self.sample_code_post(x, c)
        prior_z = self.sample_code_prior(c)
        errD_post = torch.mean(self.discriminator(torch.cat((post_z, c),1)))
        errD_prior = torch.mean(self.discriminator(torch.cat((prior_z, c),1)))
        costD = -(errD_prior - errD_post)
        costG = -costD 
        
        dec_target = response[:,1:].contiguous().view(-1)
        mask = dec_target.gt(0) # [(batch_sz*seq_len)]
        masked_target = dec_target.masked_select(mask) 
        output_mask = mask.unsqueeze(1).expand(mask.size(0), self.vocab_size)
        output = self.decoder(torch.cat((post_z, c),1), None, response[:,:-1], (res_lens-1)) 
        flattened_output = output.view(-1, self.vocab_size) 
        masked_output = flattened_output.masked_select(output_mask).view(-1, self.vocab_size)
        lossAE = self.criterion_ce(masked_output/self.temp, masked_target)
        return [('valid_loss_AE', lossAE.item()),('valid_loss_G', costG.item()), ('valid_loss_D', costD.item())]
        
    def sample(self, context, context_lens, utt_lens, floors, repeat, SOS_tok, EOS_tok):    
        self.context_encoder.eval()
        self.decoder.eval()
        
        c = self.context_encoder(context, context_lens, utt_lens, floors)
        c_repeated = c.expand(repeat, -1)
        prior_z = self.sample_code_prior(c_repeated)    
        sample_words, sample_lens= self.decoder.sampling(torch.cat((prior_z,c_repeated),1), 
                                                         None, self.maxlen, SOS_tok, EOS_tok, "greedy") 
        return sample_words, sample_lens 
      
    
    def adjust_lr(self):
        self.lr_scheduler_AE.step()
    



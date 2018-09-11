
def config_DialogWAE():
    conf = {
    'maxlen':40, # maximum utterance length
    'diaglen':10, # how many utterance kept in the context window

# Model Arguments
    'emb_size':200, # size of word embeddings
    'n_hidden':300, # number of hidden units per layer
    'n_layers':1, # number of layers
    'noise_radius':0.2, # stdev of noise for autoencoder (regularizer)
    'z_size':200, # dimension of z # 300 performs worse
    'lambda_gp':10, # Gradient penalty lambda hyperparameter.
    'temp':1.0, # softmax temperature (lower --> more discrete)
    'dropout':0.5, # dropout applied to layers (0 = no dropout)

# Training Arguments
    'batch_size':32,
    'epochs':100, # maximum number of epochs
    'min_epochs':2, # minimum number of epochs to train for

    'n_iters_d':5, # number of discriminator iterations in training
    'lr_ae':1.0, # autoencoder learning rate
    'lr_gan_g':5e-05, # generator learning rate
    'lr_gan_d':1e-05, # critic/discriminator learning rate
    'beta1':0.9, # beta1 for adam
    'clip':1.0,  # gradient clipping, max norm
    'gan_clamp':0.01,  # WGAN clamp (Do not use clamp when you apply gradient penelty             
    }
    return conf 

def config_DialogWAE_GMP():
    conf=config_DialogWAE()
    conf['n_prior_components']=3  # DailyDial 5 SWDA 3
    conf['gumbel_temp']=0.1
    return conf


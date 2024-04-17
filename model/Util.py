# -*- coding: utf-8 -*-
"""
@task: save model, load model 
@author: majing
@time: Tue Nov 10 16:29:42 2015
"""
import numpy as np
import pickle

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

############################ save model #######################################
## for GAN model ##
def save_model(f, model):
    ps = {}
    for p in model.params_gen:
        ps[p.name] = p.get_value() 
    for p in model.params_dis:
        ps[p.name] = p.get_value()   
    pickle.dump(ps, open(f, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    # print(f"Saved model parameters to {f}.")
    print("Saved model parameters to {}.".format(f))
    
def load_model(f, model):
    # ps = pickle.load(open(f, "rb"))
    ps = pickle.load(open(f, "rb"), encoding='latin1') # use encoding='latin1' to load python2 pickle in python3
    for p in model.params_gen:
        p.set_value(ps[p.name])
    for p in model.params_dis:
        p.set_value(ps[p.name])   
    # print(f"loaded model parameters from {f}.")
    return model    
    
## generator
def save_model_gen(f, model):
    ps = {}
    for p in model.params_gen:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    # print(f"Saved generator parameters to {f}.")
    
def load_model_gen(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params_gen:
        p.set_value(ps[p.name])    
    # print(f"loaded generator parameters from {f}.")
    return model

## discriminator 
def save_model_dis(f, model):
    ps = {}
    for p in model.params_dis:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    # print(f"Saved discriminator parameters to {f}.")
    
def load_model_dis(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params_dis:
        p.set_value(ps[p.name])
    # print(f"loaded discriminator parameters from {f}.")
    return model
    
# -*- coding: utf-8 -*-
"""
@object: Twitter
@task: train G/D
@author: majing
@variable: Nepoch, lr_g, lr_d, modelPath
@time: Jun 7, 2018
"""
 
# import importlib
# import sys
# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')
import os
 
import numpy as np
from numpy.testing import assert_array_almost_equal
 
import time
import datetime
import random
from Util import *
from evaluate import *
            
######################### split true/false instances ########################
def splitData_t_f(x_word, Len, y, yg, indexs_sub):
    x_word_sub, Len_sub, y_sub, yg_sub = [], [], [], []
    for i in indexs_sub:
        x_word_sub.append(x_word[i])
        Len_sub.append(Len[i])
        y_sub.append(y[i]) 
        yg_sub.append(yg[i])
    print(len(x_word_sub), len(Len_sub), len(y_sub), len(yg_sub))
    return x_word_sub, Len_sub, y_sub, yg_sub       
 
###################### pretrain individual D/G #########################
def pre_train_Generator(flag, model, x_word, indexs_sub, Len, y, yg, lr_g, Nepoch_G, modelPath):
    ## x_word, x_index: matrix
    ## y: (convert to Yg) ivector [0,1] 
    ## indexs: indexs vec for nonR or NR in train set
    # print(f"pre training Generator {flag}...")
    print("pre training Generator {}...".format(flag))
    x_word_sub, Len_sub, y_sub, yg_sub = splitData_t_f(x_word, Len, y, yg, indexs_sub)
       
    losses, num_examples_seen = [], 0
    for epoch in range(Nepoch_G):
        if epoch % 5 == 0:
           if flag == 'rn': 
              loss_gen = model.calculate_total_loss_gen_rn(x_word_sub, Len_sub, yg_sub) 
           if flag == 'nr': 
              loss_gen = model.calculate_total_loss_gen_nr(x_word_sub, Len_sub, yg_sub)    
           losses.append((num_examples_seen, loss_gen))
           Time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           # print(f"{Time}: train num=={num_examples_seen} epoch={epoch}: lossg={loss_gen}")
           print("{}: train num=={} epoch={}: lossg={}".format(Time, num_examples_seen, epoch, loss_gen))
           save_model(modelPath, model)
            
           ## change lr ##
           if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
              lr_g = lr_g * 0.5
              print("Setting gen lr to {}".format(lr_g))
           ## stop condition ##
           if epoch>10 and loss_gen<0.0001:
              break   
        ## one SGD 
        random.shuffle(indexs_sub) 
        if flag == 'rn':
           for i in indexs_sub:
               model.train_grn_pre(x_word[i], Len[i], yg[i], lr_g)
           loss_gen = model.calculate_total_loss_gen_rn(x_word_sub, Len_sub, yg_sub)
        if flag == 'nr':
           for i in indexs_sub:
               model.train_gnr_pre(x_word[i], Len[i], yg[i], lr_g)
           loss_gen =model.calculate_total_loss_gen_nr(x_word_sub, Len_sub, yg_sub)  
        num_examples_seen += len(indexs_sub)        
        # print(f"epoch={epoch}: lossg={loss_gen}")
        print("epoch={}: lossg={}".format(epoch, loss_gen))
def pre_train_Discriminator(model, x_word, y,  x_word_test, y_test, lr_d, Nepoch_D, modelPath_dis):
    ## x_word, x_index: matrix
    ## y: ivector [0, 1]
    print("pre training Discriminator ...")
    indexs = [i for i in range(len(y))]
    losses, num_examples_seen = [], 0
    for epoch in range(Nepoch_D):
        if epoch % 5 == 0:
           loss_dis = model.calculate_total_loss_dis(x_word, y) 
           losses.append((num_examples_seen, loss_dis))
           Time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           # print(f"{Time}: train num={num_examples_seen} epoch={epoch}: lossd={loss_dis}")
           print("{}: train num=={} epoch={}: lossg={}".format(Time, num_examples_seen, epoch, loss_dis))
           #res = evaluateDis(model, x_word_test, y_test)
           #print res
           save_model_dis(modelPath_dis, model)
           ## change lr ##
           if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
              lr_d = lr_d * 0.5
              # print(f"Setting dis lr to {lr_d}")
        ## one SGD 
        random.shuffle(indexs) 
        for i in indexs:
            model.train_d(x_word[i], y[i], lr_d)
        loss_dis = model.calculate_total_loss_dis(x_word, y)
        num_examples_seen += len(indexs)
        # print(f"epoch={epoch}: lossd={loss_dis}")
    #return lr_d
 
###################### train Generator & Discriminator#########################      
##  for all testing instances
def evaluateDis(model, x_word_test, Y_test):
    prediction = []
    for j in range(len(Y_test)):
        prediction.append( model.dis1(x_word_test[j]) )   
    res = evaluation_2class(prediction, Y_test) 
    return res
     
def trainRNN_step(flag, model, x_word, Len, y, yg, learning_rate):
    ## x_word, x_index: matrix
    ## y: (ground truth) ivector [0, 1] for train generator
    ## y: (convert to Yg) ivector [0, 1] for train  Discriminator
    if flag == "gen_nr":
       model.train_gnr(x_word, Len, y, learning_rate) 
       #model.train_gnr_pre(x_word, x_index, Len, y, learning_rate) 
    if flag == "gen_rn":
       model.train_grn(x_word, Len, y, learning_rate)
       #model.train_grn_pre(x_word, x_index, Len, y, learning_rate)
    if flag == "dis_nr":  
       model.train_d(x_word, y, learning_rate) # training discriminator from original non-rumor sample
       model.train_dnr(x_word, Len, yg, learning_rate) # generative sample Xnr(n -> r)
       model.train_dnrn(x_word, Len, y, learning_rate) # generative sample Xnrn(n -> r -> n)
    if flag == "dis_rn":  
       model.train_d(x_word, y, learning_rate) # training discriminator from original rumor sample
       model.train_drn(x_word, Len, yg, learning_rate) # generative sample Xrn(r -> n)
       model.train_drnr(x_word, Len, y, learning_rate) # generative sample Xrnr(r -> n -> r)
        
def train_Gen_Dis(model, x_word, Len, y, yg, index_t, index_f, x_word_test, y_test, lr_g, lr_d, Nepoch, modelPath):
    '''
    :param model: used model
    :param x_word: word frequency matrix for training set(shape: event num*vocabulary size(794*5000))
    :param Len: number of tweets in each event
    :param y: event one-hot label for training set
    :param yg: generator one-hot label(contrary to y)
    :param index_t: index of true label event(non-rumor)
    :param index_f: index of false label event(rumor)
    :param x_word_test: word frequency matrix for test set
    :param y_test: event one-hot label for test set
    :param lr_g: learning rate for generator
    :param lr_d: learning rate for discriminator
    :param Nepoch: number of epoch
    :param modelPath: used model path
    '''
    ## x_word, x_index: matrix
    ## y: (ground truth) ivector [0, 1] for train generator      
    print("training Generator & Discriminator together ...")
    print("non-rumor num:", end="\t")
    x_word_t, Len_t, y_t, yg_t = splitData_t_f(x_word, Len, y, yg, index_t) # split true label data
    print("rumor num:", end="\t")
    x_word_f, Len_f, y_f, yg_f = splitData_t_f(x_word, Len, y, yg, index_f) # split false label data
    
    indexs = index_t + index_f
    random.shuffle(indexs)  
    batchsize, f = 200, 0
    for k in range(2):
        for j in indexs:
            if j in index_t:
                # non-rumor --> rumor
                trainRNN_step("dis_nr", model, x_word[j], Len[j], y[j], yg[j], lr_d)
            if j in index_f:
                # rumor --> non-rumor
                trainRNN_step("dis_rn", model, x_word[j], Len[j], y[j], yg[j], lr_d)
 
    losses_gen, losses_dis, num_examples_seen = [], [], 0
    acc_o = 0.0
    for epoch in range(Nepoch):
        print("Epoch:", epoch)
        if epoch % 40 == 0:
           l_gnr, l_cnr, l_nr = model.calculate_total_loss_gen_cnr(x_word_t, Len_t, y_t)
           l_grn, l_crn, l_rn = model.calculate_total_loss_gen_crn(x_word_f, Len_f, y_f)
           loss_g, loss_c, loss_gen = (l_gnr + l_grn)/2, (l_cnr + l_crn)/2, (l_nr + l_rn)/2

           # discriminator loss for original data
           loss_dis_org = model.calculate_total_loss_dis(x_word, y)
           # discriminator loss for first time generated data
           loss_dis_gen = model.calculate_total_loss_dis_gen(x_word_t, Len_t, yg_t, x_word_f, Len_f, yg_f)
           # discriminator loss for reconstructed data
           loss_dis_gen2 = model.calculate_total_loss_dis_gen2(x_word_t, Len_t, y_t, x_word_f, Len_f, y_f)
           loss_dis = (loss_dis_org + loss_dis_gen + loss_dis_gen2) /3

           losses_gen.append((num_examples_seen, loss_gen))
           losses_dis.append((num_examples_seen, loss_dis))

           Time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           # print(f"{Time}: train num={num_examples_seen} epoch={epoch}: lossg={loss_gen} lossd={loss_dis}")
           # print(f"gen: (lg={loss_g} lc={loss_c}) dis: (l_o={loss_dis_org} l_g={loss_dis_gen})")
           print("{}: train num={} epoch={}: lossg={} lossd={}".format(Time, num_examples_seen, epoch, loss_gen, loss_dis))
           print("gen: (lg={} lc={}) dis: (l_o={} l_g={})".format(loss_g, loss_c, loss_dis_org, loss_dis_gen))
           #save_model(modelPath, model)
           ## test model ##
           # res = evaluateDis(model, x_word_test, y_test)
            
           ## change lr ##
           if len(losses_gen) > 1 and losses_gen[-1][1] > losses_gen[-2][1]:
              lr_g = lr_g * 0.5
              # print(f"Setting gen lr to {lr_g}")
              if lr_g < 0.00001:
                 lr_g = 0.005
           if len(losses_dis) > 1 and losses_dis[-1][1] > losses_dis[-2][1]:
              lr_d = lr_d * 0.5
              # print(f"Setting gen lr to {lr_d}")
              if lr_d < 0.00001:
                 lr_d = 0.005
           ## STOP CONDITION ##
           if epoch > 20 and abs(loss_gen-loss_dis)<0.001:
              break
        ## one SGD 
        # train generator 
        if f == 0:      
            random.shuffle(indexs)
        for k in range(1):
            random.shuffle(indexs)
            n1, n2 = 0,0
            for j in indexs[f*batchsize:(f+1)*batchsize]:
                if j in index_t:
                   trainRNN_step("gen_nr", model, x_word[j], Len[j], y[j], yg[j], lr_g) # Xnr
                   n1 += 1
                if j in index_f:
                   trainRNN_step("gen_rn", model, x_word[j], Len[j], y[j], yg[j], lr_g) # Xrn
                   n2 += 1
            #print 'G:',n1, n2           
        l_gnr, l_cnr, l_nr = model.calculate_total_loss_gen_cnr(x_word_t, Len_t, y_t)
        l_grn, l_crn, l_rn = model.calculate_total_loss_gen_crn(x_word_f, Len_f, y_f)
        loss_g, loss_c, loss_gen = (l_gnr + l_grn)/2, (l_cnr + l_crn)/2, (l_nr + l_rn)/2
        # train discriminator
        for k in range(2):
            n1, n2 = 0,0
            for j in indexs[f*batchsize:(f+1)*batchsize]:
                if j in index_t:
                   trainRNN_step("dis_nr", model, x_word[j], Len[j], y[j], yg[j], lr_d)
                   n1 += 1
                if j in index_f:            
                   trainRNN_step("dis_rn", model, x_word[j], Len[j], y[j], yg[j], lr_d)
                   n2 += 1
            #print 'D:', n1, n2
                   
        loss_dis_org = model.calculate_total_loss_dis(x_word, y)
        loss_dis_gen = model.calculate_total_loss_dis_gen(x_word_t, Len_t, yg_t, x_word_f, Len_f, yg_f)                
        loss_dis_gen2 = model.calculate_total_loss_dis_gen2(x_word_t, Len_t, y_t, x_word_f, Len_f, y_f) 
        loss_d = (loss_dis_org+loss_dis_gen+loss_dis_gen2) /3
        ## test model ##           
        res = evaluateDis(model, x_word_test, y_test)
        acc = res[1] 
        #print res[1], acc_o
        if res[1] > acc_o:
           save_model(modelPath, model)
           acc_o = res[1]
           print("new RES:", res)
        # print(f"epoch={epoch}: acc={acc} lg={loss_g} lc={loss_c} ld={loss_d}")
        #print "epoch=%d: acc=%f" % (epoch, acc)        
        num_examples_seen += n1 + n2   
        f = int((f+1) % (len(indexs)/batchsize-1))
    # final output
    model = load_model(modelPath, model)
    RES = evaluateDis(model, x_word_test, y_test)
    print("final Res:", RES)
    #return lr_g, lr_d   
   
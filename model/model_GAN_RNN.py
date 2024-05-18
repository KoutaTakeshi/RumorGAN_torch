__doc__ = """GRU-RNN aka GAN. two seq2seq-based generators and one RNN-based discriminator."""

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
import time

torch.autograd.set_detect_anomaly(True) # print the error message when backpropagation is not well defined

####### pre-defiend functions #######
def init_matrix(shape):
    return torch.randn(*shape) * 0.1

def init_vector(shape):
    return torch.zeros(*shape)
    
######################################    
class GAN(nn.Module):
    def __init__(self, vocab_size, hidden_size=5, Nclass=2, momentum=0.9):
        # self.X_word = None ## input X index
        # #self.X_index = T.imatrix('x_index')
        # self.Len = None ## generator sentence length
        # self.Y = None # ground truth
        # self.Yg = None # generated label for x
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.Nclass = Nclass
        self.momentum = momentum
        self.define_train_test_funcs()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # for param in self.params_gnr + self.params_grn + self.params_dis:
        #     param.to(self.device)

    class Generator_NR(nn.Module):
        def __init__(self, vocab_size, hidden_size=5, Nclass=2, bptt_truncate=4):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.Nclass = Nclass
            self.bptt_truncate = bptt_truncate
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # encoder embedding matrix
            self.Eg_en_nr = nn.Parameter(torch.randn(hidden_size, vocab_size))
            # decoder embedding matrix
            self.Eg_de_nr = nn.Parameter(torch.randn(vocab_size, hidden_size))
            # decoder embedding bias
            self.cg_de_nr = nn.Parameter(torch.randn(vocab_size))

            # encoder current input weight matrix
            self.Wg_en_nr = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            # encoder previous hidden state weight matrix
            self.Ug_en_nr = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            # encoder GRU bias
            self.bg_en_nr = nn.Parameter(torch.randn(3, hidden_size))

            # decoder current input weight matrix
            self.Wg_de_nr = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            # decoder previous hidden state weight matrix
            self.Ug_de_nr = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            # decoder GRU bias
            self.bg_de_nr = nn.Parameter(torch.randn(3, hidden_size))

            self.params_gnr = [self.Eg_en_nr, self.Eg_de_nr, self.cg_de_nr, self.Wg_en_nr, self.Ug_en_nr, self.bg_en_nr,
                               self.Wg_de_nr, self.Ug_de_nr, self.bg_de_nr]
            # self.params_gnr = [param.to(self.device) for param in self.params_gnr]
            self.to(self.device)

        def generate(self, x, l):
            def encode_sen_step(xt, st_prev):
                st_prev = st_prev.to(self.device)
                xe = torch.matmul(self.Eg_en_nr, xt)
                zt = torch.sigmoid(torch.matmul(self.Wg_en_nr[0], xe) + torch.matmul(self.Ug_en_nr[0], st_prev) + self.bg_en_nr[0])
                rt = torch.sigmoid(torch.matmul(self.Wg_en_nr[1], xe) + torch.matmul(self.Ug_en_nr[1], st_prev) + self.bg_en_nr[1])
                ct = torch.tanh(torch.matmul(self.Wg_en_nr[2], xe) + torch.matmul(self.Ug_en_nr[2], st_prev * rt) + self.bg_en_nr[2])
                st = zt * st_prev + (1 - zt) * ct
                return st

            # x = torch.tensor(x)
            # s_en_nr = torch.zeros(x.shape[0], self.hidden_size)
            # for i in range(x.shape[0]):
            #     s_en_nr[i] = encode_sen_step(x[i, :].flatten(), s_en_nr[i - 1] if i > 0 else torch.zeros(self.hidden_size))

            s_en_nr_list = []
            for i in range(x.shape[0]):
                s_en_nr_list.append(encode_sen_step(x[i, :].flatten(),
                                     s_en_nr_list[i - 1] if i > 0 else torch.zeros(self.hidden_size)))
            s_en_nr = torch.stack(s_en_nr_list)

            w1 = F.relu(torch.matmul(self.Eg_de_nr, s_en_nr[-1].clone()) + self.cg_de_nr) # use clone to avoid in-place operation
            
            def decode_step(wt_prev, st_prev):
                xe = torch.matmul(self.Eg_en_nr, wt_prev)
                zt = torch.sigmoid(torch.matmul(self.Wg_de_nr[0], xe) + torch.matmul(self.Ug_de_nr[0], st_prev) + self.bg_de_nr[0])
                rt = torch.sigmoid(torch.matmul(self.Wg_de_nr[1], xe) + torch.matmul(self.Ug_de_nr[1], st_prev) + self.bg_de_nr[1])
                ct = torch.tanh(torch.matmul(self.Wg_de_nr[2], xe) + torch.matmul(self.Ug_de_nr[2], st_prev * rt) + self.bg_de_nr[2])
                st = zt * st_prev + (1 - zt) * ct
                wt = F.relu(torch.matmul(self.Eg_de_nr, st) + self.cg_de_nr)
                return wt, st
            
            # words = torch.zeros(l, self.vocab_size)
            # for i in range(l):
            #     w, s_en_nr[-1] = decode_step(w1 if i == 0 else words[i - 1], s_en_nr[-1])
            #     words[i] = w

            words_list = []
            for i in range(l):
                w, s_en_nr[-1] = decode_step(w1 if i == 0 else words_list[i - 1], s_en_nr[-1].clone())
                words_list.append(w)
            words = torch.stack(words_list)
            
            return torch.cat((w1.unsqueeze(0), words[:-1]))

    class Generator_RN(nn.Module):
        def __init__(self, vocab_size, hidden_size=5, Nclass=2, bptt_truncate=4):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.Nclass = Nclass
            self.bptt_truncate = bptt_truncate
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize parameters
            self.Eg_en_rn = nn.Parameter(torch.randn(hidden_size, vocab_size))
            self.Eg_de_rn = nn.Parameter(torch.randn(vocab_size, hidden_size))
            self.cg_de_rn = nn.Parameter(torch.randn(vocab_size))
            
            self.Wg_en_rn = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            self.Ug_en_rn = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            self.bg_en_rn = nn.Parameter(torch.randn(3, hidden_size))
            
            self.Wg_de_rn = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            self.Ug_de_rn = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            self.bg_de_rn = nn.Parameter(torch.randn(3, hidden_size))

            self.params_grn = [self.Eg_en_rn, self.Eg_de_rn, self.cg_de_rn, self.Wg_en_rn, self.Ug_en_rn, self.bg_en_rn,
                               self.Wg_de_rn, self.Ug_de_rn, self.bg_de_rn]
            # self.params_grn = [param.to(self.device) for param in self.params_grn]
            self.to(self.device)

        def generate(self, x, l):
            def encode_sen_step(xt, st_prev):
                st_prev = st_prev.to(self.device)
                xe = torch.matmul(self.Eg_en_rn, xt)
                zt = torch.nn.functional.hardsigmoid(torch.matmul(self.Wg_en_rn[0], xe) + torch.matmul(self.Ug_en_rn[0], st_prev) + self.bg_en_rn[0])
                rt = torch.nn.functional.hardsigmoid(torch.matmul(self.Wg_en_rn[1], xe) + torch.matmul(self.Ug_en_rn[1], st_prev) + self.bg_en_rn[1])
                ct = torch.tanh(torch.matmul(self.Wg_en_rn[2], xe) + torch.matmul(self.Ug_en_rn[2], st_prev * rt) + self.bg_en_rn[2])
                st = zt * st_prev + (1 - zt) * ct
                return st

            # x = torch.tensor(x)
            # s_en_rn = torch.zeros(l, self.hidden_size)
            # for i in range(l):
            #     s_en_rn[i] = encode_sen_step(x[i, :].flatten(), s_en_rn[i - 1] if i > 0 else torch.zeros(self.hidden_size))

            s_en_rn_list = []
            for i in range(l):
                s_en_rn_list.append(encode_sen_step(x[i, :].flatten(),
                                    s_en_rn_list[i - 1] if i > 0 else torch.zeros(self.hidden_size)))
            s_en_rn = torch.stack(s_en_rn_list) # every time step's hidden state

            w1 = F.relu(torch.matmul(self.Eg_de_rn, s_en_rn[-1].clone()) + self.cg_de_rn)
            
            def decode_step(wt_prev, st_prev):
                xe = torch.matmul(self.Eg_en_rn, wt_prev)
                zt = torch.nn.functional.hardsigmoid(torch.matmul(self.Wg_de_rn[0], xe) + torch.matmul(self.Ug_de_rn[0], st_prev) + self.bg_de_rn[0])
                rt = torch.nn.functional.hardsigmoid(torch.matmul(self.Wg_de_rn[1], xe) + torch.matmul(self.Ug_de_rn[1], st_prev) + self.bg_de_rn[1])
                ct = torch.tanh(torch.matmul(self.Wg_de_rn[2], xe) + torch.matmul(self.Ug_de_rn[2], st_prev * rt) + self.bg_de_rn[2])
                st = zt * st_prev + (1 - zt) * ct
                wt = F.relu(torch.matmul(self.Eg_de_rn, st) + self.cg_de_rn)
                return wt, st
            
            # words = torch.zeros(l, self.vocab_size)
            # for i in range(l):
            #     w, s_en_rn[-1] = decode_step(w1 if i == 0 else words[i - 1], s_en_rn[-1])
            #     words[i] = w

            words_list = []
            for i in range(l):
                w, s_en_rn[-1] = decode_step(w1 if i == 0 else words_list[i - 1], s_en_rn[-1].clone())
                words_list.append(w)
            words = torch.stack(words_list)

            return torch.cat((w1.unsqueeze(0), words[:-1]))
            
    class Discriminator(nn.Module):
        def __init__(self, vocab_size, hidden_size=5, Nclass=2, bptt_truncate=4):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.Nclass = Nclass
            self.bptt_truncate = bptt_truncate
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize parameters
            self.Eg_en = nn.Parameter(torch.randn(hidden_size, vocab_size))
            self.Wd = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            self.Ud = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
            self.bd = nn.Parameter(torch.randn(3, hidden_size))
            self.Vd = nn.Parameter(torch.randn(Nclass, hidden_size))
            self.cd = nn.Parameter(torch.randn(Nclass))
            self.params_d = [self.Eg_en, self.Wd, self.Ud, self.bd, self.Vd, self.cd]
            # self.params_d = [param.to(self.device) for param in self.params_d]
            self.to(self.device)

        def discriminate(self, x):
            def _recurrence(xt, st_prev):
                st_prev = st_prev.to(self.device)
                xe = torch.matmul(self.Eg_en, xt)
                zt = torch.nn.functional.hardsigmoid(
                    torch.matmul(self.Wd[0], xe) + torch.matmul(self.Ud[0], st_prev) + self.bd[0])
                rt = torch.nn.functional.hardsigmoid(
                    torch.matmul(self.Wd[1], xe) + torch.matmul(self.Ud[1], st_prev) + self.bd[1])
                ct = torch.tanh(torch.matmul(self.Wd[2], xe) + torch.matmul(self.Ud[2], st_prev * rt) + self.bd[2])
                st = zt * st_prev + (1 - zt) * ct
                return st

            # x = torch.tensor(x)
            s_d_list = []
            # s_d_list.to(self.device)
            for i in range(x.shape[0]):
                s_d_list.append(_recurrence(x[i, :].flatten(),
                                s_d_list[i - 1] if i > 0 else torch.zeros(self.hidden_size)))
            s_d = torch.stack(s_d_list)

            avgS = s_d[-1]
            prediction_c = F.softmax(torch.matmul(self.Vd, avgS) + self.cd, dim=0)
            return prediction_c

        def contCmp(self, Xw, Xe):
            results = torch.mean((Xw - Xe)**2, dim=1)
            return torch.mean(results)
            
    def define_train_test_funcs(self):
        G_NR = self.Generator_NR(self.vocab_size, self.hidden_size, self.Nclass)
        G_RN = self.Generator_RN(self.vocab_size, self.hidden_size, self.Nclass)
        D = self.Discriminator(self.vocab_size, self.hidden_size, self.Nclass)
        
        self.params_dis = D.params_d
        self.params_gnr = G_NR.params_gnr
        self.params_grn = G_RN.params_grn
        self.params_gen = G_NR.params_gnr + G_RN.params_grn          
        lr = 0.0001
        
        ## step1: update generator NR
        def gen_nr(X_word, Len):
            # step 1.1: X_n -> G_nr -> X_nr
            return G_NR.generate(X_word, Len)
        self.gen_nr = gen_nr
                                        
        def d_gen_nr(X_word, Len):
            return D.discriminate(gen_nr(X_word, Len))
        self.d_gen_nr = d_gen_nr
                
        def gen_nrn(X_word, Len):
            # step 1.2: X_nr -> G_rn -> X_nrn(rumor -> non-rumor2)
            return G_RN.generate(gen_nr(X_word, Len), Len)
        self.gen_nrn = gen_nrn
        
        def f_loss_nrn(X_word, Len):
            # loss_rec: reconstruction loss
            return D.contCmp(X_word, gen_nrn(X_word, Len))
        self.f_loss_nrn = f_loss_nrn

        def loss_gen_nr(X_word, Len, Yg):
            # loss_D: discriminator loss
            return torch.sum((d_gen_nr(X_word, Len)-Yg)**2)
        self.loss_gen_nr = loss_gen_nr
    
        #loss_nr = loss_gnr + 0.02*loss_nrn 
        def f_loss_nr(X_word, Len, Yg):
            return loss_gen_nr(X_word, Len, Yg) + f_loss_nrn(X_word, Len)
        self.f_loss_nr = f_loss_nr

        # gparams_gnr_pre = [] ## only loss_D
        def train_gnr_pre(X_word, Len, Yg, lr):
            optimizer = optim.SGD(self.params_gnr, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = loss_gen_nr(X_word, Len, Yg)
            loss.backward()
            optimizer.step()
        self.train_gnr_pre = train_gnr_pre

        # gparams_gnr = []  ## loss_D +loss_rec
        def train_gnr(X_word, Len, Yg, lr):
            optimizer = optim.SGD(self.params_gen, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = f_loss_nr(X_word, Len, Yg)
            loss.backward()
            optimizer.step()
        self.train_gnr = train_gnr

        ## step2: update generator RN
        def gen_rn(X_word, Len):
            # step 2.1: X_r -> G_rn -> X_rn
            return G_RN.generate(X_word, Len)
        self.gen_rn = gen_rn
                                        
        def d_gen_rn(X_word, Len):
            return D.discriminate(gen_rn(X_word, Len))
        self.d_gen_rn = d_gen_rn
                
        def loss_gen_rn(X_word, Len, Yg):
            # loss_D
            return torch.sum((d_gen_rn(X_word, Len)-Yg)**2)
        self.loss_gen_rn = loss_gen_rn
        
        def gen_rnr(X_word, Len):
            # step 2.2: X_rn -> G_nr -> X_rnr
            return G_NR.generate(gen_rn(X_word, Len), Len)
        self.gen_rnr = gen_rnr
        
        def f_loss_rnr(X_word, Len):
            # loss_rec
            return D.contCmp(X_word, gen_rnr(X_word, Len))
        self.f_loss_rnr = f_loss_rnr
        
        #loss_rn = loss_grn + 0.02*loss_rnr 
        def f_loss_rn(X_word, Len, Yg):
            return loss_gen_rn(X_word, Len, Yg) + f_loss_rnr(X_word, Len)
        self.f_loss_rn = f_loss_rn

        # gparams_grn_pre = [] ## only loss_D
        def train_grn_pre(X_word, Len, Yg, lr):
            optimizer = optim.SGD(self.params_grn, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = loss_gen_rn(X_word, Len, Yg)
            loss.backward()
            optimizer.step()
        self.train_grn_pre = train_grn_pre

        # gparams_grn = []
        def train_grn(X_word, Len, Yg, lr):
            optimizer = optim.SGD(self.params_gen, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = f_loss_rn(X_word, Len, Yg)
            loss.backward()
            optimizer.step()
        self.train_grn = train_grn
        

        ## step3: update discriminator
        def dis1(X_word):
            # for orignal X
            return D.discriminate(X_word)
        self.dis1 = dis1
        
        def loss_dis1(X_word, Y):
            return torch.sum((dis1(X_word) - Y)**2)
        self.loss_dis1 = loss_dis1

        # gparams_d = []
        def train_d(X_word, Y, lr):
            optimizer = optim.SGD(self.params_dis, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = loss_dis1(X_word, Y)
            loss.backward()
            optimizer.step()
        self.train_d = train_d
        
        def loss_dis2(X_word, Len, Yg):
            # for X_nr
            return torch.sum((d_gen_nr(X_word, Len) - Yg)**2)
        self.loss_dis2 = loss_dis2

        # gparams_dnr = []
        def train_dnr(X_word, Len, Yg, lr):
            optimizer = optim.SGD(self.params_dis, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = loss_dis2(X_word, Len, Yg)
            loss.backward()
            optimizer.step()
        self.train_dnr = train_dnr
        
        def loss_dis3(X_word, Len, Yg):
            # for X_rn
            return torch.sum((d_gen_rn(X_word, Len) - Yg)**2)
        self.loss_dis3 = loss_dis3

        # gparams_drn = []
        def train_drn(X_word, Len, Yg, lr):
            optimizer = optim.SGD(self.params_dis, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = loss_dis3(X_word, Len, Yg)
            loss.backward()
            optimizer.step()
        self.train_drn = train_drn

        def loss_dis4(X_word, Len, Y):
            # for X_nrn
            return torch.sum((D.discriminate(gen_nrn(X_word, Len)) - Y)**2)
        self.loss_dis4 = loss_dis4

        # gparams_dnrn = []
        def train_dnrn(X_word, Len, Y, lr):
            optimizer = optim.SGD(self.params_dis, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = loss_dis4(X_word, Len, Y)
            loss.backward()
            optimizer.step()
        self.train_dnrn = train_dnrn
        
        def loss_dis5(X_word, Len, Y):
            # for X_rnr
            return torch.sum((D.discriminate(gen_rnr(X_word, Len)) - Y)**2)
        self.loss_dis5 = loss_dis5

        # gparams_drnr = []
        def train_drnr(X_word, Len, Y, lr):
            optimizer = optim.SGD(self.params_dis, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = loss_dis5(X_word, Len, Y)
            loss.backward()
            optimizer.step()
        self.train_drnr = train_drnr
        
        # gparams_dnr2 = []
        def train_dnr2(X_word, Len, Y, Yg, lr):
            optimizer = optim.SGD(self.params_dis, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = 0.3*loss_dis2(X_word, Len, Yg)+0.5*loss_dis1(X_word, Y)+0.2*loss_dis4(X_word, Len, Y)
            loss.backward()
            optimizer.step()
        self.train_dnr2 = train_dnr2
        
        # gparams_drn2 = []
        def train_drn2(X_word, Len, Y, Yg, lr):
            optimizer = optim.SGD(self.params_dis, lr=lr, momentum=self.momentum)
            optimizer.zero_grad()
            loss = 0.3*loss_dis3(X_word, Len, Yg)+0.5*loss_dis1(X_word, Y)+0.2*loss_dis5(X_word, Len, Y)
            loss.backward()
            optimizer.step()
        self.train_drn2 = train_drn2

    # def gradient_descent(self, params, gparams, learning_rate):
    #     """Momentum GD with gradient clipping."""
    #     return updates
    #     momentum_velocity_ = [0.0] * len(gparams)
    #     grad_norm = sum([torch.norm(grad)**2 for grad in gparams])**0.5
    #     updates = {}
    #     not_finite = torch.isnan(grad_norm) | torch.isinf(grad_norm)
    #     scaling_den = torch.maximum(5.0, grad_norm)
    #     for n, (param, grad) in enumerate(zip(params, gparams)):
    #         grad = torch.where(not_finite, 0.1 * param, grad * (5.0 / scaling_den))
    #         velocity = momentum_velocity_[n]
    #         update_step = self.momentum * velocity - learning_rate * grad
    #         momentum_velocity_[n] = update_step
    #         updates[param] = param + update_step
    #     return updates
        
    ##################### calculate total loss #######################    
    # only loss D
    def calculate_total_loss_gen_nr(self, xw, l, yg):
        ## x: all instances
        L = 0
        print("Calculate gen(n->r) loss:", end="\t")
        for i in tqdm(range(len(yg))):
            L += self.loss_gen_nr(xw[i], l[i], yg[i])
        return L/len(yg)

    def calculate_total_loss_gen_rn(self, xw, l, yg):
        ## x: all instances
        L = 0
        print("Calculate gen(r->n) loss:", end="\t")
        for i in tqdm(range(len(yg))):
            L += self.loss_gen_rn(xw[i], l[i], yg[i])
        return L/len(yg)
    # only loss D + loss Cont
    def calculate_total_loss_gen_cnr(self, xw, l, yg):
        ## x: all instances
        Lg, Lc, L = 0, 0, 0
        print("Calculate all gen(n->r) loss:", end="\t")
        for i in tqdm(range(len(yg))):
            L += self.f_loss_nr(xw[i], l[i], yg[i])
            Lg += self.loss_gen_nr(xw[i], l[i], yg[i])
            Lc += self.f_loss_nrn(xw[i], l[i])
        return Lg/len(yg), Lc/len(yg), L/len(yg)

    def calculate_total_loss_gen_crn(self, xw, l, yg):
        ## x: all instances
        Lg, Lc, L = 0, 0, 0
        print("Calculate all gen(r->n) loss:", end="\t")
        for i in tqdm(range(len(yg))):
            L += self.f_loss_rn(xw[i], l[i], yg[i])
            Lg += self.loss_gen_rn(xw[i], l[i], yg[i])
            Lc += self.f_loss_rnr(xw[i], l[i])
        return Lg/len(yg), Lc/len(yg), L/len(yg)

    def calculate_total_loss_dis(self, xw, y):
        ## x: all instances
        L = 0
        print("Calculate dis loss:", end="\t")
        for i in tqdm(range(len(y))):
            L += self.loss_dis1(xw[i], y[i])
        return L/len(y)

    def calculate_total_loss_dis_gen(self, xw_t, l_t, y_t, xw_f, l_f, y_f):
        ## x: all instances
        ## y -> yg
        L1 = 0
        print("Calculate dis loss for first time generated data:", end="\t")
        ## for each training instance: Xnr
        for i in tqdm(range(len(y_t))):
            L1 += self.loss_dis2(xw_t[i], l_t[i], y_t[i])
        ## for each training instance: Xrn
        for i in tqdm(range(len(y_f))):
            L1 += self.loss_dis3(xw_f[i], l_f[i], y_f[i])
        return L1/(len(y_t)+len(y_f))

    def calculate_total_loss_dis_gen2(self, xw_t, l_t, y_t, xw_f, l_f, y_f):
        ## x: all instances
        ## y -> yg
        L2 = 0
        #print len(y_t),len(xw_t), len()len(y_f)
        print("Calculate dis loss for reconstructed data:", end="\t")
        ## for each training instance: Xnr
        for i in tqdm(range(len(y_t))):
            L2 += self.loss_dis4(xw_t[i], l_t[i], y_t[i])
        ## for each training instance: Xrn
        for i in tqdm(range(len(y_f))):
            L2 += self.loss_dis5(xw_f[i], l_f[i], y_f[i])
        return L2/(len(y_t)+len(y_f))

    # def calculate_total_loss_gen_nr(self, xw, l, yg):
    #     print("Calculate gen(n->r) loss...", end="\t")
    #     L = self.loss_gen_nr(xw, l, yg).mean()
    #     return L
    #
    # def calculate_total_loss_gen_rn(self, xw, l, yg):
    #     print("Calculate gen(r->n) loss...", end="\t")
    #     L = self.loss_gen_rn(xw, l, yg).mean()
    #     return L
    #
    # def calculate_total_loss_gen_cnr(self, xw, l, yg):
    #     print("Calculate all gen(n->r) loss...", end="\t")
    #     Lg = self.loss_gen_nr(xw, l, yg).mean()
    #     Lc = self.f_loss_nrn(xw, l).mean()
    #     L = self.f_loss_nr(xw, l, yg).mean()
    #     return Lg, Lc, L
    #
    # def calculate_total_loss_gen_crn(self, xw, l, yg):
    #     print("Calculate all gen(r->n) loss...", end="\t")
    #     Lg = self.loss_gen_rn(xw, l, yg).mean()
    #     Lc = self.f_loss_rnr(xw, l).mean()
    #     L = self.f_loss_rn(xw, l, yg).mean()
    #     return Lg, Lc, L
    #
    # def calculate_total_loss_dis(self, xw, y):
    #     print("Calculate dis loss...", end="\t")
    #     L = self.loss_dis1(xw, y).mean()
    #     return L
    #
    # def calculate_total_loss_dis_gen(self, xw_t, l_t, y_t, xw_f, l_f, y_f):
    #     print("Calculate dis loss for first time generated data...", end="\t")
    #     L1 = (self.loss_dis2(xw_t, l_t, y_t) + self.loss_dis3(xw_f, l_f, y_f)).mean()
    #     return L1
    #
    # def calculate_total_loss_dis_gen2(self, xw_t, l_t, y_t, xw_f, l_f, y_f):
    #     print("Calculate dis loss for reconstructed data...", end="\t")
    #     L2 = (self.loss_dis4(xw_t, l_t, y_t) + self.loss_dis5(xw_f, l_f, y_f)).mean()
    #     return L2

    # def calculate_total_loss_gen_nr(self, xw, l, yg):
    #     start_time = time.time()
    #     print("Calculate gen(n->r) loss...", end="\t")
    #     L = self.loss_gen_nr(xw, l, yg).mean()
    #     end_time = time.time()
    #     print("Time used: {} seconds".format(end_time - start_time))
    #     return L
    #
    # def calculate_total_loss_gen_rn(self, xw, l, yg):
    #     start_time = time.time()
    #     print("Calculate gen(r->n) loss...", end="\t")
    #     L = self.loss_gen_rn(xw, l, yg).mean()
    #     end_time = time.time()
    #     print("Time used: {} seconds".format(end_time - start_time))
    #     return L
    #
    # def calculate_total_loss_gen_cnr(self, xw, l, yg):
    #     start_time = time.time()
    #     print("Calculate all gen(n->r) loss...", end="\t")
    #     Lg = self.loss_gen_nr(xw, l, yg).mean()
    #     Lc = self.f_loss_nrn(xw, l).mean()
    #     L = self.f_loss_nr(xw, l, yg).mean()
    #     end_time = time.time()
    #     print("Time used: {} seconds".format(end_time - start_time))
    #     return Lg, Lc, L
    #
    # def calculate_total_loss_gen_crn(self, xw, l, yg):
    #     start_time = time.time()
    #     print("Calculate all gen(r->n) loss...", end="\t")
    #     Lg = self.loss_gen_rn(xw, l, yg).mean()
    #     Lc = self.f_loss_rnr(xw, l).mean()
    #     L = self.f_loss_rn(xw, l, yg).mean()
    #     end_time = time.time()
    #     print("Time used: {} seconds".format(end_time - start_time))
    #     return Lg, Lc, L
    #
    # def calculate_total_loss_dis(self, xw, y):
    #     start_time = time.time()
    #     print("Calculate dis loss...", end="\t")
    #     L = self.loss_dis1(xw, y).mean()
    #     end_time = time.time()
    #     print("Time used: {} seconds".format(end_time - start_time))
    #     return L
    #
    # def calculate_total_loss_dis_gen(self, xw_t, l_t, y_t, xw_f, l_f, y_f):
    #     start_time = time.time()
    #     print("Calculate dis loss for first time generated data...", end="\t")
    #     L1 = (self.loss_dis2(xw_t, l_t, y_t) + self.loss_dis3(xw_f, l_f, y_f)).mean()
    #     end_time = time.time()
    #     print("Time used: {} seconds".format(end_time - start_time))
    #     return L1
    #
    # def calculate_total_loss_dis_gen2(self, xw_t, l_t, y_t, xw_f, l_f, y_f):
    #     start_time = time.time()
    #     print("Calculate dis loss for reconstructed data...", end="\t")
    #     L2 = (self.loss_dis4(xw_t, l_t, y_t) + self.loss_dis5(xw_f, l_f, y_f)).mean()
    #     end_time = time.time()
    #     print("Time used: {} seconds".format(end_time - start_time))
    #     return L2
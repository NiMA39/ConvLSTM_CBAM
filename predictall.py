#!/usr/bin/env python

from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import cupy
from chainer import initializers
import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import re
import glob
import cv2
from chainer import variable
from chainer import Variable
from numpy.random import *
from numpy import *
from chainer import cuda,optimizers,Chain,dataset,ChainList,training
from chainer.training import extensions
from chainer.training import triggers
xp = cuda.cupy
import chainermn
import matplotlib.pyplot as plt
from PIL import Image

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts



class ChannelAttention(Chain):
    def __init__(self, mid):
        super(ChannelAttention, self).__init__(
        fc1 = L.Convolution2D(None, mid, 1, nobias=True),
        fc2 = L.Convolution2D(None, mid, 1, nobias=True)
        )
        self.mid = mid

    def __call__(self, x):
        channel_average = F.average_pooling_2d(x,ksize=x.shape[2])#(batch,256,1,1)
        channel_max = F.max_pooling_2d(x,ksize=x.shape[2])
        channel_average = F.broadcast_to(channel_average, (x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        channel_max = F.broadcast_to(channel_max, (x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        avg_out = self.fc2(F.relu(self.fc1(channel_average)))
        max_out = self.fc2(F.relu(self.fc1(channel_max)))
        out = avg_out + max_out
        return F.sigmoid(out)

class SpatialAttention(Chain):
    def __init__(self):
        super(SpatialAttention, self).__init__(
        conv1 = L.Convolution2D(None, 1, 7, pad=3, nobias=True)
        )

    def __call__(self, x):
        spatial_average = F.average(x,axis=1,keepdims=True)
        spatial_max = F.max(x,axis=1,keepdims=True)
        spatial_concat = F.concat([spatial_average,spatial_max],axis=1)
        out = self.conv1(spatial_concat)
        out = F.broadcast_to(out, (x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        return F.sigmoid(out)


class newLSTM(Chain):
    def __init__(self,mid,sz):
        super(newLSTM, self).__init__(#(1,3,100,100)→(1,mid*4,100,100)
            Wx_Inc_1 = L.Convolution2D(None, mid, ksize=1, pad=0),
            Wx_Inc_3 = L.Convolution2D(None, mid*2, ksize=3, pad=1),
            Wx_Inc_5 = L.Convolution2D(None, mid//2, ksize=5, pad=2),
            Wx_concat = L.Convolution2D(None, mid*4, ksize=1, pad=0),
            Wx_input = L.Convolution2D(None,mid*4,ksize=sz,pad=sz//2),
            channel_attention = ChannelAttention(mid),
            spatial_attention = SpatialAttention(),
            Wh1_Linear = L.Linear(None,256),
            Wh2_Linear = L.Linear(None,256),
            Wh1_conv = L.Convolution2D(None, 1, ksize=7, pad=3),
            Wh_repeat = L.Convolution2D(None, mid*4, ksize=sz, pad=sz//2, nobias = True)#(1,3,100,100)→(1,mid*4,100,100)
        )

        self.mid = mid #64
        self.pc = None
        self.ph = None

        with self.init_scope():
            Wci_initializer = initializers.Zero()
            self.Wci = variable.Parameter(Wci_initializer)
            Wcf_initializer = initializers.Zero()
            self.Wcf = variable.Parameter(Wcf_initializer)
            Wco_initializer = initializers.Zero()
            self.Wco = variable.Parameter(Wco_initializer)


    def set_state(self, pc = None, ph = None):
        self.pc = pc
        self.ph = ph

    def initialize_params(self,x):
        self.Wci.initialize((self.mid,x.shape[2],x.shape[3]))
        self.Wcf.initialize((self.mid,x.shape[2],x.shape[3]))
        self.Wco.initialize((self.mid,x.shape[2],x.shape[3]))

    def initialize_state(self,x):
        self.pc = Variable(self.xp.zeros((x.shape[0],self.mid,x.shape[2],x.shape[3]), dtype = self.xp.float32))
        self.ph = Variable(self.xp.zeros((x.shape[0],self.mid,x.shape[2],x.shape[3]), dtype = self.xp.float32))

    def __call__(self, x):
        if self.Wci.data is None:
            self.initialize_params(x)

        if self.pc is None:
            self.initialize_state(x)

#        Inc1 = self.Wx_Inc_1(x)#64
 #       Inc3 = self.Wx_Inc_3(x)#128
 #       Inc5 = self.Wx_Inc_5(x)#32
 #       maxpool = F.max(x,axis=1,keepdims=True)#(1,3,100,100)→(1,1,100,100)
 #       Inc_concat = F.concat([Inc1,Inc3,Inc5,maxpool],axis=1)
 #       Inc_output = self.Wx_concat(Inc_concat)#(100,100,256)#(1,256,100,100)
        Inc_output = self.Wx_input(x)
        #CBAM
        out = self.channel_attention(self.ph) * self.ph
        out = self.spatial_attention(out) * out
        ch_repeat_gate = self.ph + out
        ch_repeat_gate = self.Wh_repeat(ch_repeat_gate)

        ci = F.sigmoid(Inc_output[:,0:self.mid,:,:] + ch_repeat_gate[:,0:self.mid,:,:] + F.scale(self.pc, self.Wci, 1))
        cf = F.sigmoid(Inc_output[:,self.mid:self.mid*2,:,:] + ch_repeat_gate[:,self.mid:self.mid*2,:,:] +F.scale(self.pc, self.Wcf, 1))
        ca = F.tanh(Inc_output[:,self.mid*2:self.mid*3,:,:] + ch_repeat_gate[:,self.mid*2:self.mid*3,:,:])
        cc = cf * self.pc + ci * ca
        co = F.sigmoid(Inc_output[:,self.mid*3:self.mid*4,:,:] + ch_repeat_gate[:,self.mid*3:self.mid*4,:,:] + F.scale(cc, self.Wco, 1))
        ch = co * F.tanh(cc)

        self.pc = cc
        self.ph = ch

        return ch


class MLP(chainer.Chain):

    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():

            self.Enc_lstm1 = newLSTM(128,5)
            self.Enc_lstm2 = newLSTM(64,5)
            self.Enc_lstm3 = newLSTM(64,5)
            self.Dec_lstm1 = newLSTM(128,5)
            self.Dec_lstm2 = newLSTM(64,5)
            self.Dec_lstm3 = newLSTM(64,5)
            self.last = L.Convolution2D(None,1,1)

    def __call__(self, x):#x.shape = (1,4,1,64,64)
    #    print(x.shape)
        k0 = x.shape[1]
        k1 = x.shape[3]
        k2 = x.shape[4]
        self.Enc_lstm1.set_state()
        self.Enc_lstm2.set_state()
        self.Enc_lstm3.set_state()
        for i in range(k0):
            x1 = x[:,i,:,:]#対応する矩形(1,1,64,64)
            #print(x1.shape)
            h1 = self.Enc_lstm1(x1)
            h2 = self.Enc_lstm2(h1)
            h3 = self.Enc_lstm3(h2)

        self.Dec_lstm1.set_state(self.Enc_lstm1.pc, self.Enc_lstm1.ph)
        self.Dec_lstm2.set_state(self.Enc_lstm2.pc, self.Enc_lstm2.ph)
        self.Dec_lstm3.set_state(self.Enc_lstm3.pc, self.Enc_lstm3.ph)

        ans_list = []
        for i in range(10):
            if i == 0:
                h1 = self.Dec_lstm1(Variable(self.xp.zeros((x.shape[0],1,k1,k2), dtype = self.xp.float32)))
            else:
                h1 = self.Dec_lstm1(ans)
            h2 = self.Dec_lstm2(h1)
            h3 = self.Dec_lstm3(h2)
            h = F.concat((h1, h2, h3))
            ans = F.sigmoid(self.last(h))
            if i == 0:
                ans1 = ans
            else:
                ans1 = F.concat((ans1,ans))
    #    print(ans.shape)

        return ans1


    def get_loss_func(self):

        def loss_func(x,y):
            pred_y = self.__call__(x)
            #print("x.shape, y.shape, pred_y.shape", x.shape, y.shape, pred_y.shape)
            #print("type x y predy", type(x), type(y), type(pred_y))
            # self.loss = F.mean_squared_error(self.__call__(x), y)
            self.loss = F.sum(-(F.log(pred_y) * y + F.log(1-pred_y) * (1-y)))
#            print("loss", self.loss)
            #print("y[0,0]", y[0,0])
#            print("pred_y[0,0]", pred_y[0,0])

            chainer.report({'cross_entropy': self.loss}, observer=self)
            return self.loss

        return loss_func


    #     h1 = self.Dec_lstm1(Variable(self.xp.zeros((x.shape[0],1,k1,k2), dtype = self.xp.float32)))
    #     h2 = self.Dec_lstm2(h1)
    #     h3 = self.Dec_lstm3(h2)
    #     h = F.concat((h1, h2, h3))
    #     ans = self.last(h)
    # #    print(ans.shape)
    #
    #     return ans
    #
    #
    # def get_loss_func(self):
    #
    #     def loss_func(x,y):
    #         self.loss = F.mean_squared_error(self.__call__(x), y)
    #         chainer.report({'mse': self.loss}, observer=self)
    #         return self.loss
    #
    #     return loss_func


gpu=0
model = MLP()
if gpu >= 0:
    chainer.cuda.get_device_from_id(gpu).use()
    model.to_gpu()

chainer.serializers.load_npz('./mymodel.npz',model)
print("model type is ", type(model))
data = np.load("./mnist_test_seq.npy")
l = 7000
ind = l
for k in range(10):
    values = data[:10, ind, :, :].astype(np.float32)#(10,64,64)
    input_all = []
    for i in range(10):
        value = values[i,:,:].astype(np.uint8)
        input_all.append(value)
        #pil_img = Image.fromarray(value)
        #pil_img.save("./resultimg1/input"+str(ind)+"_"+str(i)+".png")
    input_result = np.concatenate([i for i in input_all],axis=1)
    pil_img1 = Image.fromarray(input_result)
    pil_img1.save("./resultimg/inputall"+str(ind)+".png")
    values = np.expand_dims(values,axis=1)#(10,1,64,64)
    values /= 255
    labels = data[10:20, ind, :, :]
    values = np.expand_dims(values,axis=0)#(1,10,1,64,64)
    values = cupy.asarray(values)
    print("x[0,0]",values[0])
    print(values.shape)
    y = model(values)#(1,10,64,64)
    print("y[0,0]",y[0,0])
    y = cupy.asnumpy(y.data)
    print(y.shape)
    print(type(y))
    output = np.squeeze(y,axis=0)
    output_all = []
    label_all = []
    for i in range(10):
        output_dims = output[i,:,:]#(64,64)
        label = labels[i,:,:]
        print(output_dims.shape)
        img = np.asarray(output_dims)
        img *= 255
        print("show the predict image")
        img = img.astype(np.uint8)
        output_all.append(img)
        label_all.append(label)
     #   pil_img1 = Image.fromarray(img)
     #   pil_img2 = Image.fromarray(label)
     #   pil_img1.save("./resultimg1/result"+str(ind)+"_"+str(i)+".png")
     #   pil_img2.save("./resultimg1/groundtruth"+str(ind)+"_"+str(i)+".png")

    output_result = np.concatenate([i for i in output_all],axis=1)
    label_result = np.concatenate([i for i in label_all],axis=1)
    print(output_result.shape)
    pil_img2 = Image.fromarray(label_result)
    pil_img2.save("./resultimg/labelall"+str(ind)+".png")
    pil_img3 = Image.fromarray(output_result)
    pil_img3.save("./resultimg/outputall"+str(ind)+".png")
    ind = ind + 1


print("end")
ind = 7000
loss = 0
for i in range(3000):
    values = data[:10, ind+i, :, :].astype(np.float32)#(10,64,64)
    values = np.expand_dims(values,axis=1)#(10,1,64,64)
    values /= 255
    labels = data[10:20, ind+i, :, :].astype(np.float32)#(10,64,64)
    labels /= 255
    labels = np.expand_dims(labels,axis=0)#(1,10,64,64)
    values = np.expand_dims(values,axis=0)#(1,10,1,64,64)
    values = cupy.asarray(values)
    #print("x[0,0]",values[0])
    #print(values.shape)
    y = model(values)#y = (1,10,64,64)
    #print("y[0,0]",y[0,0])
    y = cupy.asnumpy(y.data)
    #print(y.shape)
    #print(type(y))

    loss += np.sum(np.sum(-(np.log(y) * labels + np.log(1-y) * (1-labels)),axis=2),axis=2)#(1,10,64,64)->(1,10)
    #loss = F.sum(-(F.log(pred_y) * y + F.log(1-pred_y) * (1-y)))

cross_entropy_all = []
for i in range(10):
    print("predict_"+str(i+1)+" cross_entropy is ",loss[:,i]/3000)
    cross_entropy_all.append(loss[:,i]/3000)

np.save("./resultimg/cross_entropy_all",cross_entropy_all)

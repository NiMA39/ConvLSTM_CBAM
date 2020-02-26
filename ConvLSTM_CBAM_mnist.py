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

    def __call__(self, x):#(1,1,64,64)        if self.Wci.data is None:
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
        print(ans1.shape)
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


class MovingMnistDataset(chainer.dataset.DatasetMixin):
    def __init__(self, l, r, path="./mnist_test_seq.npy"):
        self.l = l
        self.r = r
        self.data = np.load(path)


    def __len__(self):
        return self.r - self.l

    def get_example(self, i):
        ind = self.l + i
        values = self.data[:10, ind, :, :].astype(np.float32)
        values = np.expand_dims(values,axis=1)
        values /= 255
        label = self.data[10:20, ind, :, :].astype(np.float32)
        label /= 255

        return values,label


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=102400,
                        help='Number of units')
    parser.add_argument('--inf', type=int, default=10)
    parser.add_argument('--outf', type=int, default=10)
#    parser.add_argument('--communicator', type=str,default='hierarchical', help='Type of communicator')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

#    # Setup an optimizer
#    optimizer = chainer.optimizers.Adam()
#    optimizer.setup(model)
#
    #VGG = MLP(args.unit,101)
    #model = L.Classifier(VGG)
    model = MLP()
#    print(model.forward_lstm1.Wxi.W.data)
#    print(model.params())
#    print(sum(p.data.size for p in model.params()))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    # train = ImageTrainDataset_UCF101()
     # train_iter = chainer.iterators.SerialIterator(train,args.batchsize)
    train = MovingMnistDataset(0, 7000)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batchsize, shuffle=True)
    test = MovingMnistDataset(7000, 10000)
    test_iter = chainer.iterators.SerialIterator(test, batch_size=args.batchsize, repeat=False, shuffle=False)

    #print(test_iter[0])
    # Setup an optimizer
    #optimizer = chainer.optimizers.Adam()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
#    model.vgg.disable_update()
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu, loss_func=model.get_loss_func())
    stop_trigger = (args.epoch,'epoch')
    trainer = training.Trainer(updater,stop_trigger,out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu, eval_func=model.get_loss_func()), name='val')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch','main/cross_entropy','elapsed_time','val/main/cross_entropy']))
    trainer.extend(extensions.ProgressBar())
    log_interval = (1,'epoch')
    trainer.extend(extensions.snapshot(),trigger=log_interval)
    trainer.extend(extensions.snapshot_object(model, filename='model_epoch-{.updater.epoch}'))
#    print(sum(p.data.size for p in model.params()))
    #trainer.extend(extensions.snapshot_object(model,'model_iter_{.updater.iteration}'),trigger=log_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume,trainer)

    trainer.run()

    chainer.serializers.save_npz("./result/mymodel.npz", model)



if __name__ == '__main__':
    main()
        # ch_repeat_gate = self.Wh_repeat(self.ph)#(1,256,100,100)
        #
        # channel_average = F.average_pooling_2d(F.relu(ch_repeat_gate),ksize=100)
        # channel_max = F.max_pooling_2d(F.relu(ch_repeat_gate),ksize=100)
        # channel_average = self.Wh2_Linear(self.Wh1_Linear(channel_average))
        # channel_average = F.expand_dims(channel_average,axis=2)
        # channel_average = F.expand_dims(channel_average,axis=2)#(1, 256, 1, 1)
        #
        #
        # channel_max = self.Wh2_Linear(self.Wh1_Linear(channel_max))
        # channel_max = F.expand_dims(channel_max,axis=2)
        # channel_max = F.expand_dims(channel_max,axis=2)#(1, 256, 1, 1)
        #
        # CBAM_channel = F.sigmoid(channel_average + channel_max)#(1,256,1,1)
        #
        # CBAM_feature = CBAM_channel ch_repeat_gate

        # spatial_average = F.average(CBAM_feature,axis=1,keepdims=True)#(1,1,100,100)
        # spatial_max = F.max(CBAM_feature,axis=1,keepdims=True)#(1,1,100,100)
        # spatial_concat = F.concat([spatial_average,spatial_max],axis=1)#(1,2,100,100)
        #

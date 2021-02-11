import time
from math import sqrt

import pandas as pd
import scipy.sparse as sparse

import torch
import model
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

def customedMSEloss(inputs, targets, size_average=False):
  mask = targets != 0
  num_ratings = torch.sum(mask.float())
  criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
  return criterion(inputs * mask.float(), targets), Variable(torch.Tensor([1.0])) if size_average else num_ratings

# check GPU
use_gpu = torch.cuda.is_available() # global flag
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if use_gpu:
    print('GPU is available.')
    torch.cuda.set_device(device)
else:
    print('GPU is not available.')

# Load train data
one_hot_encoding = pd.read_pickle('one_hot_array_ply.pkl')
one_hot_encoding = one_hot_encoding.tolil()

# Parameter
layer_size = one_hot_encoding.shape[1]
hidden_layers = "2048,1024" # 24684
non_linearity_type = "selu"
drop_prob = 0.8
weight_decay = 0.0001 # L2 규제
lr = 0.0005
num_epochs = 50
aug_step = 1
batch_size = 2048

# Dataloader
n_songs = one_hot_encoding.shape[0]
rep = n_songs//batch_size

dataloader = [one_hot_encoding[i*batch_size:(i+1)*batch_size] for i in range(rep+1)]

# AutoEncoder Model
rencoder = model.AutoEncoder(layer_sizes=[layer_size] + [int(l) for l in hidden_layers.split(',')],
                               nl_type= non_linearity_type, dp_drop_prob= drop_prob)


# train
optimizer = optim.Adam(rencoder.parameters(),lr= lr, weight_decay=weight_decay)

Loss = nn.MSELoss()
w1, w2 = 15, 200

for epoch in range(1,num_epochs+1):
    print('Doing epoch {} of {}'.format(epoch, num_epochs))
    e_start_time = time.time()
    rencoder.train()
    total_epoch_loss = 0.0
    denom = 0.0

    for i in range(rep+1):
        mb = torch.tensor(dataloader[i].todense(),dtype = torch.float)
        inputs = Variable(mb)
        optimizer.zero_grad()
        outputs = rencoder(inputs)

        loss1, num_ratings = customedMSEloss(outputs, inputs) # num_ratings ; 원래 1이었던 값의 수. masked MSE. 원래 값이 있던 것에 대한 loss만 계산
        loss1 = loss1 / num_ratings
        loss2 = Loss(inputs, outputs) # 0인 부분을 모두 1로 맞추지 않도록
        loss = loss1*w1 + loss2*w2

        if i%5 == 0:
            print("curr loss :",loss.item(), f"   =>   1: {loss1.item()*w1:.3f} , 2: {loss2.item()*w2:.3f}")
            print("  - examples : ",outputs[0].detach())
            print(f"  - {outputs[0].max().item():.3f}, {outputs[0].min().item():.3f}")

        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()
        denom += 1

        if epoch > 30 and aug_step > 0 and i > 0:
            # Magic data augmentation trick happen here
            for t in range(aug_step):
                inputs = Variable(outputs.data)
                optimizer.zero_grad()
                outputs = rencoder(inputs)
                loss1, num_ratings = model.MSEloss(outputs, inputs) # num_ratings ; 원래 1이었던 값의 수
                loss1 = loss1 / num_ratings
                loss2 = Loss(inputs, outputs)
                loss = loss1 + loss2*w2
                loss.backward()
                optimizer.step()

    e_end_time = time.time()
    print('Total epoch {} finished in {} minutes with TRAINING RMSE loss: {}'
        .format(epoch, (e_end_time - e_start_time)//60, sqrt(total_epoch_loss/denom)))
    if epoch % 15 == 0 or epoch == num_epochs:
        print("Saving model to {}".format("auto"+ "_epoch"+str(epoch))+"_l2_3.pt")
        torch.save(rencoder.state_dict(), "song_embedding_selu_"+ "_epoch"+str(epoch)+".pt")
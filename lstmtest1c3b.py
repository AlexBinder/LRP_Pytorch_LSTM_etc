
import os #anyone remember 5 1/4" soft diskettes? 

from typing import List, Tuple, Optional, overload

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

import torch
import torch.nn as nn
from torch import Tensor




from lrp_general6 import oneparam_wrapper_class,linearlayer_eps_wrapper_fct
from lrp_pytorchlstm1 import *




# the actual prediction model,
# what needs to be wrapped for an explanation: the LSTM cell for each layer inside self.rnn
# and the nn.Linear self.fc


# no bias in LSTM
class lstmsubmodel(nn.Module):

  def __init__(self , device):
    super(lstmsubmodel,self).__init__()

    self.indim =2
    self.hiddendim =1 # was 2
    self.num_layers=1
    self.rnn =  nn.LSTM(input_size = self.indim, hidden_size = self.hiddendim, num_layers = self.num_layers, bias = False,  batch_first = False, dropout = 0, bidirectional = False  )
    self.fc = nn.Linear(self.hiddendim,1, bias = False)

    self.device=device

  def forward(self,bsize,xt):
    h0 = torch.zeros( (self.num_layers, bsize, self.hiddendim) , device = self.device)
    c0 = torch.zeros( (self.num_layers, bsize, self.hiddendim)  , device = self.device )
    _,(hn2, cn2) =  self.rnn.forward( xt, (h0,c0 ) ) 
    #print('hn2.shape' , hn2.shape)
    y = self.fc( hn2[-1,:,:]) # trivial for one layer

    return y #, hn2, cn2


# use bias in LSTM
class lstmsubmodel2(nn.Module):

  def __init__(self , device):
    super(lstmsubmodel2,self).__init__()

    self.indim =2
    self.hiddendim =1 # was 2
    self.num_layers=1
    self.rnn =  nn.LSTM(input_size = self.indim, hidden_size = self.hiddendim, num_layers = self.num_layers, bias = True,  batch_first = False, dropout = 0, bidirectional = False  )
    self.fc = nn.Linear(self.hiddendim,1, bias = False)

    self.device=device

  def forward(self,bsize,xt):
    h0 = torch.zeros( (self.num_layers, bsize, self.hiddendim) , device = self.device)
    c0 = torch.zeros( (self.num_layers, bsize, self.hiddendim)  , device = self.device )
    _,(hn2, cn2) =  self.rnn.forward( xt, (h0,c0 ) ) 
    #print('hn2.shape' , hn2.shape)
    y = self.fc( hn2[-1,:,:]) # trivial for one layer

    return y #, hn2, cn2


# this wraps nn.LSTM, see runtest_explain2(...) on how to use it
def wraplstm(indim, hiddendim, num_layers, trainedmodel , lrp_params):

  assert isinstance( trainedmodel, nn.LSTM)

  eps = lrp_params['lstm_eps']

  replacementrnn = lstm_unidir_bsany(indim, hiddendim, num_layers, bias = False)
  for k in range(0,num_layers):

    name='weight_ih_l{}'.format(k)
    weight_ih= getattr(trainedmodel, name).clone()
    name='weight_hh_l{}'.format(k)
    weight_hh= getattr(trainedmodel, name).clone()

    name='bias_ih_l{}'.format(k)
    if hasattr(trainedmodel, name):
      bias_ih = getattr(trainedmodel, name).clone()
    else:
      bias_ih = None

    name='bias_hh_l{}'.format(k)
    if hasattr(trainedmodel, name):
      bias_hh = getattr(trainedmodel, name).clone()
    else:
      bias_hh = None
  
    # set biases later
    if (bias_ih is None) and (bias_hh is None):
      celllayer= lstmcellsurrogate(indim, hiddendim, bias = False)
      celllayer.setfrom_nobias( weight_ih, weight_hh)
    else:
      celllayer= lstmcellsurrogate(indim, hiddendim, bias = True)
      celllayer.setfrom_havebias( weight_ih, weight_hh, bias_ih, bias_hh)
      

    replacementrnn.lstmcell[k]=lstmcell_wrapper_class( celllayer , lstmcell_wrapper_fct(), eps)

  return replacementrnn
  # need also rule for fc layer 

#wraps a nn.Linear
def wrapfc(module, lrp_params):

  assert isinstance(module, nn.Linear)
  linearwrapper = oneparam_wrapper_class(module, autogradfunction= linearlayer_eps_wrapper_fct(),
                                      parameter1=lrp_params['linear_eps'])

  return linearwrapper




##############################################
##############################################
##############################################
# dataloader, train, and testing code,
##############################################
##############################################
##############################################



def unidirlstmtest3nobias():
  #test to  get forwardpass right, no bias case

  #init hidden state, random x, check predictions in eval mode

  indim = 10
  hiddendim = 13

  numtest = 100

  diff1=0.
  diff2=0.

  diff1a=0.
  diff2a=0.

  num_layers = 3
  bsize=17
  for t in range(numtest):

    randlen = torch.randint( low = 2, high = 10, size = (1,) )
    #print('randlen.shape', randlen.shape)

    #randlen=[1]

    x = torch.rand ( (randlen[0], bsize, indim) )*10-5
    h0 = torch.rand ( (num_layers, bsize, hiddendim) )*10-5 #torch.zeros( ( bsize, hiddendim) ) # torch.rand ( ( 1, hiddendim) )*10-5
    c0 = torch.rand ( (num_layers, bsize, hiddendim) )*10-5 #torch.zeros( ( bsize, hiddendim) ) # torch.rand ( ( 1, hiddendim) )*10-5
    
    ownlstm = lstm_unidir_bsany(indim, hiddendim, num_layers = num_layers, bias = False)


    defaultlstm = nn.LSTM(input_size = indim, hidden_size = hiddendim, num_layers = num_layers, bias = False,  batch_first = False, dropout = 0, bidirectional = False  )



    #copy weights over

    for l in range(num_layers):

      name='weight_ih_l{}'.format(l)
      weight_ih = getattr(defaultlstm, name).clone()
      name='weight_hh_l{}'.format(l)
      weight_hh = getattr(defaultlstm, name).clone()
      ownlstm.lstmcell[l].setfrom_nobias( weight_ih, weight_hh)

    _, (hn, cn) = ownlstm.forward( x.clone(), (h0.clone(),c0.clone()) )
    _,(hn2, cn2) = defaultlstm.forward( x.clone(), (h0.clone() ,c0.clone() ) ) # unsqueeze is for numlayers*numdirs


    #print(hn.shape,hn2.squeeze(0).shape)

    diff1a+= torch.mean( (hn2[0,:,:]-hn[0,:,:])**2 ).item() / numtest
    diff2a+= torch.mean( (cn2[0,:,:]-cn[0,:,:])**2 ).item() / numtest

    diff1+= torch.mean( (hn2-hn)**2 ).item() / numtest
    diff2+= torch.mean( (cn2-cn)**2 ).item() / numtest

    #print(o.shape)
  print( ' diffs_a ', diff1a, diff2a  )
  print( ' diffs ', diff1, diff2  )
  print('done')





def unidirlstmtest3bias():
  #test to get forwardpass right, case with bias

  #init hidden state, random x, check predictions in eval mode

  indim = 10
  hiddendim = 13

  numtest = 100

  diff1=0.
  diff2=0.

  diff1a=0.
  diff2a=0.

  num_layers = 3
  bsize=17
  for t in range(numtest):

    randlen = torch.randint( low = 2, high = 10, size = (1,) )
    #print('randlen.shape', randlen.shape)

    #randlen=[1]

    x = torch.rand ( (randlen[0], bsize, indim) )*10-5
    h0 = torch.rand ( (num_layers, bsize, hiddendim) )*10-5 #torch.zeros( ( bsize, hiddendim) ) # torch.rand ( ( 1, hiddendim) )*10-5
    c0 = torch.rand ( (num_layers, bsize, hiddendim) )*10-5 #torch.zeros( ( bsize, hiddendim) ) # torch.rand ( ( 1, hiddendim) )*10-5
    
    ownlstm = lstm_unidir_bsany(indim, hiddendim, num_layers = num_layers, bias = True)


    defaultlstm = nn.LSTM(input_size = indim, hidden_size = hiddendim, num_layers = num_layers, bias = True,  batch_first = False, dropout = 0, bidirectional = False  )



    #copy weights over

    for l in range(num_layers):

      name='weight_ih_l{}'.format(l)
      weight_ih = getattr(defaultlstm, name).clone()
      name='weight_hh_l{}'.format(l)
      weight_hh = getattr(defaultlstm, name).clone()

      name='bias_ih_l{}'.format(l)
      if not hasattr(defaultlstm, name):
        print( 'this lstms seems not to have field: ', name )
        exit()
      bias_ih = getattr(defaultlstm, name).clone()

      name='bias_hh_l{}'.format(l)
      if not hasattr(defaultlstm, name):
        print( 'this lstms seems not to have field: ', name )
        exit()
      bias_hh = getattr(defaultlstm, name).clone()

      #print(t,l , 'b mean' , torch.mean(bias_ih + bias_hh) )

      ownlstm.lstmcell[l].setfrom_havebias( weight_ih, weight_hh, bias_ih, bias_hh )



    _, (hn, cn) = ownlstm.forward( x.clone(), (h0.clone(),c0.clone()) )
    _,(hn2, cn2) = defaultlstm.forward( x.clone(), (h0.clone() ,c0.clone() ) ) # unsqueeze is for numlayers*numdirs



    #print(hn.shape,hn2.squeeze(0).shape)

    diff1a+= torch.mean( (hn2[0,:,:]-hn[0,:,:])**2 ).item() / numtest
    diff2a+= torch.mean( (cn2[0,:,:]-cn[0,:,:])**2 ).item() / numtest

    diff1+= torch.mean( (hn2-hn)**2 ).item() / numtest
    diff2+= torch.mean( (cn2-cn)**2 ).item() / numtest

    #print(o.shape)
  print( ' diffs_a ', diff1a, diff2a  )
  print( ' diffs ', diff1, diff2  )
  print('done')






class subtractiondataloader():


  def __init__(self, seqlen,num, bsize):
    
    assert isinstance(seqlen,tuple)


    self.lenmin=seqlen[0]
    self.lenmax=seqlen[1]
    assert isinstance(self.lenmin,int)
    assert isinstance(self.lenmax,int)

    if self.lenmin>self.lenmax:
      print('wrong')
      exit()
    if self.lenmin<2:
      print('wrong')
      exit()

    self.num=num

    self.bsize=bsize

  #def __len__(self):
  #  return self.num

    self.bcount=0

  def __iter__(self):
   
    return self

  def __next__(self):

    if self.bcount >= self.num // self.bsize:

      self.bcount=0
      #return #raise StopIteration?
      raise StopIteration

    self.bcount+=1

    #draw length
    curlen=torch.randint(low=self.lenmin, high=self.lenmax+1, size=(1,))

    # (seq_len, batch, input_size)
    xts= torch.zeros((curlen,self.bsize,2 ) )
    res= torch.zeros((self.bsize, ) )

    for b in range(self.bsize):

      #draw positions
      ind1=torch.randint(low=0, high=curlen.item()-1, size=(1,))
      ind2=torch.randint(low=ind1.item()+1, high=curlen.item(), size=(1,))

      #draw numbers
      values=  torch.rand(size=(curlen.item(),2),dtype=torch.float)
      values[:,0]=values[:,0]*0.5-1 #neg
      values[:,1]=values[:,1]*0.5+0.5 #pos

      posorneg= torch.randint(low=0,high=2,size=(curlen.item(),),dtype=torch.int)
      

      xt=torch.zeros((curlen,2), dtype=torch.float)

      for i in range(curlen):
        if (i == ind1) or (i == ind2):
          xt[i,0]=values[i, posorneg[i] ]
        else:    
          xt[i,1]=values[i, posorneg[i] ]

      xts[:,b,:]=xt
      res[b]= xt[ind1,0]-xt[ind2,0]
    #return xt, xt[ind1,0]-xt[ind2,0]

    return xts, res



class subtractiondataloader2():


  def __init__(self, seqlen,num, bsize):
    # no neg values
    assert isinstance(seqlen,tuple)


    self.lenmin=seqlen[0]
    self.lenmax=seqlen[1]
    assert isinstance(self.lenmin,int)
    assert isinstance(self.lenmax,int)

    if self.lenmin>self.lenmax:
      print('wrong')
      exit()
    if self.lenmin<2:
      print('wrong')
      exit()

    self.num=num

    self.bsize=bsize

  #def __len__(self):
  #  return self.num

    self.bcount=0

  def __iter__(self):
   
    return self

  def __next__(self):

    if self.bcount >= self.num // self.bsize:

      self.bcount=0
      #return #raise StopIteration?
      raise StopIteration

    self.bcount+=1

    #draw length
    curlen=torch.randint(low=self.lenmin, high=self.lenmax+1, size=(1,))

    # (seq_len, batch, input_size)
    xts= torch.zeros((curlen,self.bsize,2 ) )
    res= torch.zeros((self.bsize, ) )

    for b in range(self.bsize):

      #draw positions
      ind1=torch.randint(low=0, high=curlen.item()-1, size=(1,))
      ind2=torch.randint(low=ind1.item()+1, high=curlen.item(), size=(1,))

      #draw numbers
      values=  torch.rand(size=(curlen.item(),2),dtype=torch.float)
      values[:,0]=values[:,0]*0.5+0.5 #neg
      values[:,1]=values[:,1]*0.5+0.5 #pos

      posorneg= torch.randint(low=0,high=2,size=(curlen.item(),),dtype=torch.int)

      xt=torch.zeros((curlen,2), dtype=torch.float)

      for i in range(curlen):
        if (i == ind1) or (i == ind2):
          xt[i,0]=values[i, posorneg[i] ]
        else:    
          xt[i,1]=values[i, posorneg[i] ]

      xts[:,b,:]=xt
      res[b]= xt[ind1,0]-xt[ind2,0]
    #return xt, xt[ind1,0]-xt[ind2,0]

    return xts, res



def train_epoch(loadertr, model, optimizer, scheduler, device):

  loss0 = nn.MSELoss()

  model.train()
  for xt,lb in loadertr:
    xt=xt.to(device)
    lb=lb.to(device)
    bsize = xt.shape[1]

    '''
    def closure():
      optimizer.zero_grad()
      pred = model(bsize,xt)
      lossval = loss0(pred, lb.unsqueeze(1))
      lossval.backward()
      for p in model.parameters():
        n= torch.norm(p.grad.data)
        if n >5:
          p.grad.data=p.grad.data/n*5
      return lossval
    optimizer.step(closure)
    '''
    optimizer.zero_grad()
    pred=model(bsize,xt)
    lossval = loss0(pred,lb.unsqueeze(1)) 
    lossval.backward()
    for p in model.parameters():
      n= torch.norm(p.grad.data)
      if n >5:
        p.grad.data=p.grad.data/n*5
    optimizer.step()
    

  scheduler.step() 
  
def eval_epoch(loaderval, model, device ):

  loss = nn.MSELoss()

  model.eval()
  avgmseloss=0
  with torch.no_grad():
    for i ,(xt,lb) in enumerate(loaderval):
      xt=xt.to(device)
      lb=lb.to(device)
      bsize = xt.shape[1]
      pred=model(bsize,xt)
      lossval = loss(pred,lb.unsqueeze(1)) 
      avgmseloss += lossval
      ct=i
    avgmseloss/=float(ct+1)

  return avgmseloss


def train(loadertr,loaderval, model, optimizer, scheduler , device, filepath):

  numepoch=1000

  model.to(device)

  bestloss= None
  for e in range(numepoch):
    train_epoch(loadertr,model, optimizer, scheduler , device)
    avgmseloss = eval_epoch(loaderval, model , device)

    print('epoch',e,'avgmseloss',avgmseloss.item() )
    savemodel=False
    if bestloss is None:
      savemodel=True
      bestloss= avgmseloss
    elif bestloss > avgmseloss:
      savemodel=True
      bestloss= avgmseloss

    if savemodel:
      if not os.path.isdir(filepath):
        os.makedirs(filepath)
      fn = os.path.join( filepath, 'somelstm{}.pt'.format(0))
      torch.save(model.state_dict(),fn)


def testmodel(loaderte, model,device, filepath):
  fn = os.path.join( filepath, 'somelstm{}.pt'.format(0))
  dc=torch.load(fn)
  model.load_state_dict(dc)
  model.to(device)

  loss=eval_epoch(loaderte, model,device)

  return loss




def compare_outputs(loaderval, model1, model2, device ):



  model1.eval()
  model2.eval()

  diffs=0
  ct=-1
  with torch.no_grad():
    for i ,(xt,lb) in enumerate(loaderval):
      ct=i
      xt=xt.to(device)
      lb=lb.to(device)
      bsize = xt.shape[1]
      pred1 =model1(bsize,xt)
      pred2 =model2(bsize,xt)

      df=torch.mean((pred1-pred2)**2)
      #print('df',df.item())
      diffs+=df
  return diffs/float(ct+1)




def explainsomesample( model, device, loaderte ):

  for i ,(xt,lb) in enumerate(loaderte):
    xt=xt.to(device)
    lb=lb.to(device)
    bsize = xt.shape[1]

    xt.requires_grad=True
  
    pred = model(bsize, xt)

    print('pred.shape', pred.shape) # (bsize,1)
    #torch.sum(hn2).backward()
    torch.sum(pred).backward()

    print('input')
    print(xt.data.data.cpu().numpy())
    print('relevance scores')
    print(xt.grad.data.cpu().numpy())


### bias



 
def runtraintest2(bias):

  loadertr= subtractiondataloader2( ( 4,10 ) ,10000,8) #3
  loaderval= subtractiondataloader2( ( 11,12 ) ,5000,8)
  loaderte= subtractiondataloader2( ( 13,14 ) ,5000,8)

  filepath='./lstmlrptest1c3bias'

  device=torch.device('cuda:0')

  if False==bias:
    lstmused=lstmsubmodel(device)
  else:
    lstmused=lstmsubmodel2(device)

  #optimizer= torch.optim.RMSprop(lstmused.parameters(), lr=0.002 )#, weight_decay=1e-5) # good
  #optimizer= torch.optim.SGD(lstmused.parameters(), lr=0.001 )#, weight_decay=5e-6)
  #optimizer= torch.optim.AdamW(lstmused.parameters(), lr=0.002 , weight_decay= 2.5e-6) #good with 2 hidden units
  optimizer= torch.optim.AdamW(lstmused.parameters(), lr=0.002) 
  #optimizer= torch.optim.LBFGS(lstmused.parameters(), lr=0.002 )#, weight_decay= 2.5e-6)

  scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.25, last_epoch=-1)

  train(loadertr,loaderval, lstmused , optimizer, scheduler, device, filepath)



  tloss=testmodel(loaderte, lstmused, device, filepath)

  print('loss',tloss.item())


def runtest2(bias):

  loaderte= subtractiondataloader2( ( 13,14 ) ,5000,8)

  filepath='./lstmlrptest1c3bias'

  device=torch.device('cuda:0')

  if False==bias:
    lstmused=lstmsubmodel(device)
  else:
    lstmused=lstmsubmodel2(device)

  los=testmodel(loaderte, lstmused,device, filepath)
  print('test mse',los.item())







def runtest_explain0_2(bias):

  loaderte= subtractiondataloader2( ( 13, 14 ), 5000, 8)
  device=torch.device('cpu')
  filepath='./lstmlrptest1c3bias'
  fn = os.path.join( filepath, 'somelstm{}.pt'.format(0))
  dc=torch.load(fn)


  if False==bias:
    lstmused=lstmsubmodel(device)
    lstmused_sav=lstmsubmodel(device)
  else:
    lstmused=lstmsubmodel2(device)
    lstmused_sav=lstmsubmodel2(device)

  lstmused_sav.load_state_dict(dc)
  lstmused.load_state_dict(dc)

  lstmused.to(device)
  lstmused_sav.to(device)


  indim = lstmused.rnn.input_size
  hiddendim = lstmused.rnn.hidden_size 
  num_layers= lstmused.rnn.num_layers

  lrp_params= dict({'lstm_eps': 1e-3, 'linear_eps': 1e-3 })

  lstmsurrogate = wraplstm(indim, hiddendim, num_layers, trainedmodel= lstmused.rnn, lrp_params = lrp_params)

  lstmused.rnn = lstmsurrogate.to(device)

  wrappedfc= wrapfc(lstmused.fc , lrp_params = lrp_params )
  lstmused.fc = wrappedfc

  diffs=compare_outputs(loaderte, lstmused_sav , lstmused, device ) #compare if it still does the same after wrapping
  print('diffs.item()',  diffs.item() )




def runtest_explain2(bias):

  loaderte= subtractiondataloader2( ( 13, 14 ), 1, 1)

  device=torch.device('cpu')
  filepath='./lstmlrptest1c3bias'
  fn = os.path.join( filepath, 'somelstm{}.pt'.format(0))
  dc=torch.load(fn)

  if False==bias:
    lstmused=lstmsubmodel(device)
  else:
    lstmused=lstmsubmodel2(device)

  lstmused.load_state_dict(dc)
  lstmused.to(device)

  indim = lstmused.rnn.input_size
  hiddendim = lstmused.rnn.hidden_size 
  num_layers= lstmused.rnn.num_layers

  lrp_params= dict({'lstm_eps': 1e-3, 'linear_eps': 1e-3 })

  lstmsurrogate = wraplstm(indim, hiddendim, num_layers, trainedmodel= lstmused.rnn, lrp_params = lrp_params)

  lstmused.rnn = lstmsurrogate

  wrappedfc= wrapfc(lstmused.fc , lrp_params = lrp_params )
  lstmused.fc = wrappedfc

  explainsomesample( lstmused, device, loaderte )









if __name__=='__main__':

  #several testing routines

  #runtraintest()
  #runtest()
  #runtest_explain0()
  #runtest_explain()


  #several testing routines for case with bias


  # no training, just reproducibility of own lstmcell surrogate
  #unidirlstmtest3nobias()
  #unidirlstmtest3bias()

  # training and explaining
  havebias=False

  # training of model
  #runtraintest2(havebias)

  #does the wrapper reproduce the same scores?
  #runtest_explain0_2(havebias)

  # explaining a sample
  runtest_explain2(havebias)


from typing import List, Tuple, Optional, overload

import torch
import torch.nn as nn
#import torch.nn.functional as F

from torch import Tensor

import os


from lrp_general6 import lrp_backward






##############################################
##############################################
##############################################
# LSTMcell and unidir LSTM
##############################################
##############################################
##############################################



# the class of an LSTM cell for processing a single x_t and updating c_{t-1}, h_{t-1} into c_t,h_t
class lstmcellsurrogate(nn.LSTMCell):

  def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
    super(lstmcellsurrogate, self).__init__(input_size, hidden_size, bias) 

    if bias == False:
      biasterm1 = None
      biasterm2 = None
      biasterm3 = None
      biasterm4 = None
    else:
      biasterm1=torch.zeros((hidden_size,)) 
      biasterm2=torch.zeros((hidden_size,)) 
      biasterm3=torch.zeros((hidden_size,)) 
      biasterm4=torch.zeros((hidden_size,)) 

    w_ih_ingate, w_ih_forgetgate, w_ih_cellgate, w_ih_outgate = self.weight_ih.chunk(4, 0)
    w_hh_ingate, w_hh_forgetgate, w_hh_cellgate, w_hh_outgate = self.weight_hh.chunk(4, 0)

    self.xhlinear_ingate =   xhlinear ( w_ih_ingate, w_hh_ingate, biasterm1 )
    self.xhlinear_forgetgate =   xhlinear ( w_ih_forgetgate, w_hh_forgetgate, biasterm2 )
    self.xhlinear_cellgate =   xhlinear (w_ih_cellgate, w_hh_cellgate, biasterm3 ) # need relflow here
    self.xhlinear_outgate =   xhlinear (w_ih_outgate, w_hh_outgate, biasterm4 )

  def setfrom_nobias(self, weight_ih: Tensor, weight_hh: Tensor):

    self.weight_ih = torch.nn.Parameter(weight_ih)
    self.weight_hh = torch.nn.Parameter(weight_hh)

    biasterm1 = None
    biasterm2 = None
    biasterm3 = None
    biasterm4 = None

    w_ih_ingate, w_ih_forgetgate, w_ih_cellgate, w_ih_outgate = self.weight_ih.chunk(4, 0) 
    w_hh_ingate, w_hh_forgetgate, w_hh_cellgate, w_hh_outgate = self.weight_hh.chunk(4, 0)

    self.xhlinear_ingate =   xhlinear ( w_ih_ingate, w_hh_ingate, biasterm1 )
    self.xhlinear_forgetgate =   xhlinear ( w_ih_forgetgate, w_hh_forgetgate, biasterm2 )
    self.xhlinear_cellgate =   xhlinear (w_ih_cellgate, w_hh_cellgate, biasterm3 ) # need relflow here
    self.xhlinear_outgate =   xhlinear (w_ih_outgate, w_hh_outgate, biasterm4 )


  def setfrom_havebias(self, weight_ih: Tensor, weight_hh: Tensor , bias_ih: Tensor, bias_hh: Tensor):

    self.weight_ih = torch.nn.Parameter(weight_ih)
    self.weight_hh = torch.nn.Parameter(weight_hh)

    w_ih_ingate, w_ih_forgetgate, w_ih_cellgate, w_ih_outgate = self.weight_ih.chunk(4, 0) 
    w_hh_ingate, w_hh_forgetgate, w_hh_cellgate, w_hh_outgate = self.weight_hh.chunk(4, 0)

    if bias_ih is not None:
      b_ih_ingate, b_ih_forgetgate, b_ih_cellgate, b_ih_outgate = bias_ih.chunk(4, 0) 
    if bias_hh is not None:
      b_hh_ingate, b_hh_forgetgate, b_hh_cellgate, b_hh_outgate = bias_hh.chunk(4, 0) 

    if (bias_ih is not None) and (bias_hh is not None):
      biasterm1 =  torch.nn.Parameter((b_ih_ingate + b_hh_ingate).clone())
      biasterm2 =  torch.nn.Parameter((b_ih_forgetgate + b_hh_forgetgate).clone())
      biasterm3 =  torch.nn.Parameter((b_ih_cellgate + b_hh_cellgate).clone())
      biasterm4 =  torch.nn.Parameter((b_ih_outgate + b_hh_outgate).clone())
    elif (bias_ih is None) and (bias_hh is not None):
      biasterm1 =  torch.nn.Parameter(( b_hh_ingate).clone())
      biasterm2 =  torch.nn.Parameter(( b_hh_forgetgate).clone())
      biasterm3 =  torch.nn.Parameter(( b_hh_cellgate).clone())
      biasterm4 =  torch.nn.Parameter(( b_hh_outgate).clone())
    elif (bias_ih is not None) and (bias_hh is None):
      biasterm1 =  torch.nn.Parameter(( b_ih_ingate).clone())
      biasterm2 =  torch.nn.Parameter(( b_ih_forgetgate).clone())
      biasterm3 =  torch.nn.Parameter(( b_ih_cellgate).clone())
      biasterm4 =  torch.nn.Parameter(( b_ih_outgate).clone())
    else:
      biasterm1 = None
      biasterm2 = None
      biasterm3 = None
      biasterm4 = None

    self.xhlinear_ingate =   xhlinear ( w_ih_ingate, w_hh_ingate, biasterm1)
    self.xhlinear_forgetgate =   xhlinear ( w_ih_forgetgate, w_hh_forgetgate, biasterm2 )
    self.xhlinear_cellgate =   xhlinear (w_ih_cellgate, w_hh_cellgate, biasterm3 ) # need relflow here
    self.xhlinear_outgate =   xhlinear (w_ih_outgate, w_hh_outgate, biasterm4 )


  def getparams(self):
    return  self.weight_ih, self.weight_hh, self.xhlinear_ingate.bias, self.xhlinear_forgetgate.bias, self.xhlinear_cellgate.bias, self.xhlinear_outgate.bias 

  def forward(self, xt: Tensor, hc: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:


    self.check_forward_input(xt)
    if hc is None:
      bsize = xt.size(0)
      zeros = torch.zeros(bsize, self.hidden_size, dtype=xt.dtype, device=xt.device)
      hc = (zeros, zeros.clone())  # h and c or c and h ?
    self.check_forward_hidden(xt, hc[0], '[0]')
    self.check_forward_hidden(xt, hc[1], '[1]')

    #xt.shape=  (bsize, input_size)  input_size=input_dim
    h=hc[0] # (bsize, self.hidden_size) 
    c=hc[1] # (bsize, self.hidden_size) 
    
    ingate = torch.sigmoid(     self.xhlinear_ingate    (xt,h) )
    cellgate = torch.tanh(      self.xhlinear_cellgate  (xt,h) ) #ahhh, that one needs to be wrapped
    forgetgate = torch.sigmoid( self.xhlinear_forgetgate(xt,h) )

    cy = ingate * cellgate + forgetgate * c 

    outgate = torch.sigmoid(    self.xhlinear_outgate   (xt,h))
    hy = outgate * torch.tanh(cy)

    hcret = torch.cat( (hy,cy), dim=1 )
    return hcret#hy,cy
       

# the class of an unidirectional LSTM for processing a whole sequence (x_t)_{t=1}^T and returning a sequence of all (h_t,c_t)_{t=1}^T
class lstm_unidir_bsany(nn.Module):

  # accepts only sequences of equal length, for now no packedSeq
  def __init__(self, input_size: int, hidden_size: int, num_layers: int , bias: bool = True) -> None:
    super(lstm_unidir_bsany, self).__init__()

    self.indim = input_size
    self.hiddendim = hidden_size
    self.bias = bias 
    self.num_layers = num_layers

    self.lstmcell=[None for _ in range(self.num_layers)]

    self.lstmcell[0]= lstmcellsurrogate( self.indim, self.hiddendim, self.bias )
    for k in range(1,self.num_layers):
      self.lstmcell[k]= lstmcellsurrogate( self.hiddendim, self.hiddendim, self.bias )

  def forward(self, x, h0c0tuple):
    # x.shape (seq_len, batch, input_size)
    assert len(x.shape) == 3

    assert x.shape[2]   == self.indim

    seqlen = x.shape[0]
    bsize = x.shape[1]

    assert (h0c0tuple is None) or isinstance(h0c0tuple,tuple)
    if h0c0tuple is not None:
      assert len(h0c0tuple)==2

      assert h0c0tuple[0].shape[0] == self.num_layers
      assert h0c0tuple[1].shape[0] == self.num_layers

      assert h0c0tuple[0].shape[1] == x.shape[1] 
      assert h0c0tuple[1].shape[1] == x.shape[1] 

      assert h0c0tuple[0].shape[2] == self.hiddendim
      assert h0c0tuple[1].shape[2] == self.hiddendim
   
      h= h0c0tuple[0].clone() #( self.num_layers ,bsize, self.hiddendim)
      c=  h0c0tuple[1].clone() #( self.num_layers ,bsize, self.hiddendim)

      hc = torch.cat((h,c),dim=2) 
    else:

      hc = torch.zeros( (self.num_layers ,bsize, 2*self.hiddendim ), dtype=x.dtype, device=x.device)

    hlastlayer = torch.zeros(( seqlen , bsize, self.hiddendim) , dtype=x.dtype, device=x.device ) #(seq_len, batch, num_directions * hidden_size)
    for t in range(0, seqlen):
      hi,ci = torch.chunk(hc[0,:,:].clone(),2, dim=1) # chunks along 2*hiddendim
      hc[0,:,:] = self.lstmcell[0].forward( x[t,:,:], (hi, ci) )


      for k in range(1, self.num_layers):
        hi,ci = torch.chunk(hc[k,:,:].clone(),2, dim=1)  # chunks along 2*hiddendim
        hc[k,:,:] = self.lstmcell[k].forward( hc[k-1,:,:self.hiddendim], (hi, ci) )
        if k+1 == self.num_layers:
          hlastlayer[t,:,:] = hc[k,:,:self.hiddendim].clone()
    
    h,c = torch.chunk(hc,2, dim=2)
    return hlastlayer,(h,c) # for compat with pytorch lstm


##############################################
##############################################
##############################################
# some helper functions for LRP
# all using eps-rule, see examples in lrp_general6.py for how to extend it to beta-rule or other rules
##############################################
##############################################
##############################################


class xhlinear(nn.Module):
  def __init__(self, w_ih: Tensor, w_hh: Tensor, bias=None):
    super(xhlinear, self).__init__()

    #w_ih.shape=  (hidden_size, input_size)
    #w_hh.shape=  (hidden_size, hidden_size)

    self.W_xthenh= torch.nn.Parameter( torch.cat( (w_ih, w_hh), dim = 1 ) ) #( hidden_size, input_size + hidden_size) 
    if bias is None:   
      self.bias=bias
    else:
      self.bias=torch.nn.Parameter(bias)

  def forward(self,x: Tensor,h: Tensor) -> Tensor:
    xh = torch.cat( (x, h) , dim = 1 ) 
    return nn.functional.linear(xh, self.W_xthenh, self.bias ) #shape= (bsize,hidden_size)

class xhlinear_wrapper_class(nn.Module):
    def __init__(self, w_ih: Tensor, w_hh: Tensor, bias, autogradfunction, eps: float):
        super(xhlinear_wrapper_class, self).__init__()

        assert isinstance(autogradfunction, xhlinear_eps_wrapper_fct)

        self.W_xthenh = torch.nn.Parameter( torch.cat( (w_ih, w_hh), dim = 1 ) ) #( hidden_size, input_size + hidden_size)    
        if bias is None:   
          self.bias=bias
        else:
          self.bias=torch.nn.Parameter(bias)

        self.eps=eps
        self.wrapper = autogradfunction

    def forward(self, x: Tensor, h: Tensor):
        y = self.wrapper.apply(x, h,  self.W_xthenh , self.bias, self.eps)
        return y

#autogradfct for xhlinear
class xhlinear_eps_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, x, h, W_xthenh , bias , eps):

        epstensor = torch.tensor([eps], dtype=torch.float32, device=x.device)
        dim1of_x_shapetensor = torch.tensor([x.shape[1]], dtype=torch.long, device=x.device)
        xh = torch.cat( (x, h), dim = 1 ) 
        if bias is not None:
            bias = bias.data.clone()
        ctx.save_for_backward(xh, W_xthenh.data.clone(), bias, epstensor, dim1of_x_shapetensor )#,*values)  # *values unpacks the list

        return nn.functional.linear(xh, W_xthenh, bias )

    @staticmethod
    def backward(ctx, grad_output):

        xh, W_xthenh, bias, epstensor, dim1of_x_shapetensor = ctx.saved_tensors

        if bias is None:
            module = nn.Linear(in_features = W_xthenh.shape[1] , out_features = W_xthenh.shape[0] , bias=False)
        else:
            module = nn.Linear(in_features = W_xthenh.shape[1] , out_features = W_xthenh.shape[0] , bias=True)
            module.bias = torch.nn.Parameter(bias)
        module.weight = torch.nn.Parameter(W_xthenh)

        eps = epstensor.item()
        X = xh.clone().detach().requires_grad_(True)
        R = lrp_backward(_input=X, layer=module,
                         relevance_output=grad_output[0], eps0=eps, eps=eps)

        Rx = R[:, :dim1of_x_shapetensor[0].item() ]
        Rh = R[:, dim1of_x_shapetensor[0].item(): ] 

        return Rx, Rh, None, None, None


#autogradfct + wrapper for w1 * x1 + w2 * x2

class w1cellw2cellgatesum_wrapper_class(nn.Module):
    def __init__(self, autogradfunction, eps):
        super(w1cellw2cellgatesum_wrapper_class, self).__init__()

        self.eps=eps
        self.wrapper = autogradfunction

    def forward(self, w1: Tensor, cell: Tensor, w2: Tensor , cellgate: Tensor):
        y = self.wrapper.apply(w1, cell, w2 , cellgate,  self.eps)
        return y

class w1cw2cgatehelper(nn.Module):
    def __init__(self,w1,w2):
        super(w1cw2cgatehelper, self).__init__()
        self.w1=torch.nn.Parameter(w1)
        self.w2=torch.nn.Parameter(w2)
    def forward(self,cellcellgate):
      hiddensize= cellcellgate.shape[1]//2
      return self.w1*cellcellgate[:,:hiddensize] + self.w2*cellcellgate[:,hiddensize:] 

class w1cellw2cellgatesum_eps_wrapper_fct(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, w1, cell, w2 , cellgate, eps):
        epstensor = torch.tensor([eps], dtype=torch.float32, device=cell.device)
        ctx.save_for_backward(w1.data.clone(), cell.data.clone(), w2.data.clone() , cellgate.data.clone(), epstensor )
        return w1*cell+w2*cellgate

    @staticmethod
    def backward(ctx, grad_output):

        print('w1cellw2cellgatesum_eps_wrapper_fct bw')

        w1, cell, w2 , cellgate, epstensor = ctx.saved_tensors

        eps = epstensor.item()
        module = w1cw2cgatehelper( torch.nn.Parameter(w1) , torch.nn.Parameter(w2) )
        X = torch.cat((cell,cellgate),dim=1)

        X = X.clone().detach().requires_grad_(True)
        R = lrp_backward(_input=X, layer=module,
                         relevance_output=grad_output[0], eps0=eps, eps=eps)

        print('xhlinear_eps_wrapper_fct R', R.shape)
        # exit()
        Rcell = R[:, :cell.shape[1] ]
        Rcellgate = R[:, cell.shape[1]: ] 


        return None, Rcell, None, Rcellgate, None


##############################################
##############################################
##############################################
# the actual lrp wrappers for a LSTM cell
##############################################
##############################################
##############################################


class lstmcell_wrapper_class(nn.Module):
    def __init__(self, module, autogradfunction, eps):
        super(lstmcell_wrapper_class, self).__init__()

        assert isinstance(module, lstmcellsurrogate)
        assert isinstance(autogradfunction, lstmcell_wrapper_fct)

        self.module = module
        self.wrapper = autogradfunction
      
        self.eps = eps

    def forward(self, xt: Tensor, hc: Tuple[Tensor, Tensor]):
        #hnew, cnew = self.wrapper.apply(xt, hc[0], hc[1],  self.module, self.eps)
        #return hnew, cnew
        hcret = self.wrapper.apply(xt, hc[0], hc[1],  self.module, self.eps)
        return hcret


class lstmcell_wrapper_fct(torch.autograd.Function):  # to be used with generic_activation_pool_wrapper_class(module,this)
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, xt, h, c, module, eps):

        epstensor = torch.tensor([eps], dtype=torch.float32,
                                 device=xt.device)
        # get parameter values of cell

        if True == module.bias:
          ctx.save_for_backward(xt,h,c, module.weight_ih.data.clone(), module.weight_hh.data.clone(), module.xhlinear_ingate.bias.data.clone(), module.xhlinear_forgetgate.bias.data.clone(), module.xhlinear_cellgate.bias.data.clone(),  epstensor) #no need for module.xhlinear_outgate.bias,
        else:
          ctx.save_for_backward(xt,h,c, module.weight_ih.data.clone(), module.weight_hh.data.clone(), None, None, None,  epstensor)

        hcret = module.forward(xt, (h,c) )
        return hcret

    @staticmethod
    def backward(ctx, grad_output):
        print('lstmcell_wrapper_fct len(grad_output)',len(grad_output),grad_output[0].shape)
        xt,h,cell, weight_ih, weight_hh, ingate_bias, forgetgate_bias, cellgate_bias,  epstensor = ctx.saved_tensors

        bsize = xt.shape[0]
        input_size = xt.shape[1] #(bsize, self.input_size)
        hidden_size = h.shape[1] #(bsize, self.hidden_size)

        eps = epstensor.item()

        #first: Rcy_internal = (grad_output[0] + grad_output[1]).clone().detach() # if we would have h,c as return values,
        # but instead we have as return value:     hcret = torch.cat( (hy,cy), dim=1 )
        if len(grad_output[0].shape)>1:
          print('wtf lstmcell_wrapper_fct backward -> len(grad_output[0].shape)', len(grad_output[0].shape) )
          exit()

        hiddendim = grad_output[0].shape[0]//2
        Rcy_internal = (grad_output[0][ :hiddendim ] + grad_output[0][ hiddendim: ]).clone().detach()

        #second: Rcy_internal (eps on eltwise) -> R( cellgate), R( c) as weighted sum!!!, not as self.elt
        # here a backward needed

        w_ih_ingate, w_ih_forgetgate, w_ih_cellgate, _ = weight_ih.chunk(4, 0) # yes 9 #w_ih_outgate 
        w_hh_ingate, w_hh_forgetgate, w_hh_cellgate, _ = weight_hh.chunk(4, 0) #w_hh_outgate 

        xhlinear_ingate =   xhlinear ( w_ih_ingate, w_hh_ingate, ingate_bias ) # except for cell gate could use nn.Functional.Linear instead
        xhlinear_forgetgate =   xhlinear ( w_ih_forgetgate, w_hh_forgetgate, forgetgate_bias ) # except for cell gate could use nn.Functional.Linear instead
        xhlinear_cellgate = xhlinear_wrapper_class(w_ih_cellgate, w_hh_cellgate, cellgate_bias, xhlinear_eps_wrapper_fct() ,eps ) 

        ingate = torch.sigmoid(     xhlinear_ingate    (xt,h) )
        forgetgate = torch.sigmoid( xhlinear_forgetgate(xt,h) )
        cellgate = torch.tanh(  xhlinear_cellgate.forward (xt,h) ) #ahhh, that one needs to be wrapped

        w1cellw2cellgatesum = w1cellw2cellgatesum_wrapper_class ( w1cellw2cellgatesum_eps_wrapper_fct(), eps)

        forgetgate.requires_grad_(False)
        ingate.requires_grad_(False)
        cell=cell.detach().requires_grad_(True)
        cellgate= cellgate.detach().requires_grad_(True)
        with torch.enable_grad():
          cy=w1cellw2cellgatesum.forward( forgetgate, cell, ingate , cellgate )
          cy.backward( Rcy_internal.unsqueeze(0) )  # \sum_d  Rcy_internal[d] * D(cy)/ D (cell,cellgate)_d
        # have now        
        #cell.grad
        #cellgate.grad
        Rc= cell.grad.data.clone()
    
        # third: R_cellgate (eps on cellgate) -> Rx, Rh
        # here a backward
        xt = xt.clone().detach().requires_grad_(True)
        h = h.clone().detach().requires_grad_(True)
        with torch.enable_grad():
          precellgatevalue = xhlinear_cellgate.forward (xt,h)
          precellgatevalue.backward( cellgate.grad.data.clone() )
      
        Rx =  xt.grad.data.clone()
        Rh =   h.grad.data.clone()
        return Rx, Rh, Rc, None, None




################################
################################
################################
################################
################################
################################





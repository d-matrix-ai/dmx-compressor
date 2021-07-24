import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


__ALL__ = [
    "softmax"
]

def poly2softmax(x, dim=-1):
    with torch.no_grad():
        n = x.shape[dim]
        eps=1.e-30  #small number to avoid dividing by zero 
        ln2 = 0.69315  # log(2)
        invln2 = 1.4427  # 1 / log(2)
        scale = 14  # for exp() poly approximation assume 16bit integer arithmetic 
                    # with 14 bit fractional part
        c0int = 2**scale  # poly coefficient c0 = 1
        c1int = round(1.0151 * 2**scale)  # poly coefficient c1
        # note poly coefficient c2 is 0.5 

        # y = torch.zeros_like(x) # initialize output vector

        # range reduction to range -log(2)/2 < r < log(2)/2
        k = torch.round(x * invln2)
        r = x - k * ln2

        # compute exp(r) emulating fixed point arithmetic 
        rint = torch.round(r * 2**scale)
        # r2int = torch.round(rint * rint * 2**(-scale))
        # mult_add1 = torch.round(c0int + torch.round(c1int * rint * 2**(-scale)))
        # mult_add2 = torch.round(mult_add1 + 0.5 * r2int)   
        mult_add1 = torch.round(c1int + 0.5*rint)
        mult_add2 = torch.round(c0int + torch.round(mult_add1*rint*2**(-scale)))

        y = mult_add2 * 2**(-scale)

        # shift by k-kmax bits for final result 
        kmax,_ = torch.max(k,dim=-1)
        kmax = torch.unsqueeze(kmax,dim=-1)
        y *= 2**(k-kmax)

        # compute softmax (note will need to use 32 bits for sum in HW)
        sum_exp = torch.sum(y, dim=dim, keepdim=True)
        y /= (sum_exp + eps)
        # y = torch.round(y * 256) / 256
        
    return y

def base2softmax(x, dim=-1):
    with torch.no_grad():
        #This function computes Softmax using base2exp
        #function for the exp(x)
        eps=1.e-30  #small number to avoid dividing by zero 
        #compute exp(x) for input vector x
        #including integer vector k for re-normalization
        ey,k = base2exp(x,dim=dim)
  
        kmax,_ = torch.max(k,dim=-1) #find max k along softmax dim.
        kmax = torch.unsqueeze(kmax,dim=-1)

        #compute sum with normalization for numerical stability
        ey = ey*2**(k-kmax)
        sum_ey = torch.sum(ey, dim=dim, keepdim=True)

        #compute softmax 
        y=ey/(sum_ey+eps)

    return y

def base2exp(x, dim=-1):
    with torch.no_grad():
        #This function computes exp(x) using a range reduction
        #technique exp(x)=(2^k)*2^v, where k is an integer and 0<v<1
        #2^v is approximated by a simple linear interpolation d+v.
        #input x should be a vector shape (n,1)
        
        scale=10 #bits after binary fixed point
        log2e_fp=1.4426950408889634; #log2(e) in floating point 
        log2e=round(log2e_fp*2**scale)/2**scale #log2(e) 
        d=0.957 #minmax solution for d over the input range 0<v<1
        d=round(d*2**scale)/2**scale
        
        #range reduction to k and v
        z=x*log2e
        k=torch.floor(z)
        v=z-k
        v=torch.round(v*2**scale)/2**scale

        #compute exp(x)
        two_pow_v = v + d
        # import pdb
        # pdb.set_trace()    
        #ey=2**k*two_pow_v
        ey = two_pow_v
    return ey,k

def softmax(x, dim=-1):
    _x = F.softmax(x, dim=dim)
    #print(_x)
    #_x.data = poly2softmax(x, dim=dim)
    _x.data = base2softmax(x, dim=dim)
    #print(_x)
    # import pdb
    # pdb.set_trace()
    return _x


# class Softmax(Function):
#     r"""
#     d-MATRiX custom softmax in pytorch
#     """

#     @staticmethod
#     def forward(ctx, input, dim):
#         return poly2softmax(input, dim=dim), F.softmax(input, dim=dim)

#     @staticmethod
#     def backward(ctx, _g, g_output):
#         return g_output, None

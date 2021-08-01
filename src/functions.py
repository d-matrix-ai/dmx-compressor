import torch
import torch.nn.functional as F


__ALL__ = [
    "ApproximationMixin",
]


class ApproximationMixin:
    r"""
    Mixin for modules with approximated forward logic
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.approximation_function = None
        self.approximation_error = None

    def _forward(self, input, *args, **kwargs):
        _output = super().forward(input)
        if self.approximation_function is not None:
            with torch.no_grad():
                _approx = eval(self.approximation_function)(input, *args, **kwargs)
                self.approximation_error = _approx - _output.data
                _output.data = _approx
        return _output

    def extra_repr(self):
        return (
            None
            if self.approximation_function is None
            else f"approximation_function = {self.approximation_function}"
        )


# def poly2softmax(x, dim=-1, **kwargs):
#     r"""
#     This function computes Softmax using a range
#     reduction technique and second order polynomial approximation
#     for exp(r), where r is in reduced range
#     """
#     eps = 1.0e-30  # small number to avoid dividing by zero

#     #compute exp(r) and k, where exp(x)=2^k * exp(r)
#     ey,k = poly2exp(x,dim=dim)

#     kmax, _ = torch.max(k, dim=dim, keepdim=True)  # find max k along softmax dim.
#     # shift by k-kmax bits for final result
#     ey = ey * 2 ** (k - kmax)

#     # compute softmax (note will need to use 32 bits for sum in HW)
#     sum_exp = torch.sum(ey, dim=dim, keepdim=True)
#     y = ey/(sum_exp + eps)

#     return y

# def poly2exp(x, dim=-1):
#     r"""
#     This function computes exp(x) using a range reduction
#     technique x = r + k*log(2), where k is an integer and 
#     -log(2)/2 < r < +log(2)/2.  exp(x) = 2^k * exp(r),
#     where exp(r) is approximated by a second degree polynomial
#     """
#     ln2 = 0.69315  # log(2)
#     invln2 = 1.4427  # 1 / log(2)
#     scale = 14  # emulate bits after binary fixed point
#     c0int = 2**scale  # poly coefficient c0 = 1
#     c1int = round(1.0151 * 2**scale)  # poly coefficient c1
#     # note poly coefficient c2 is 0.5 

    # # range reduction to range -log(2)/2 < r < log(2)/2
    # k = torch.round(x * invln2)
    # r = x - k * ln2

    # # compute exp(r) emulating fixed point arithmetic 
    # rint = torch.round(r * 2**scale)
        
    # mult_add1 = torch.round(c1int + 0.5*rint)
    # mult_add2 = torch.round(c0int + torch.round(mult_add1*rint*2**(-scale)))
    # ey = mult_add2 * 2**(-scale) #convert back to decimal number
    
    # return ey, k

def poly2softmax(x, dim=-1, **kwargs):
    r"""
    This function computes Softmax using a range
    reduction technique and second order polynomial approximation
    for exp(r), where r is in reduced range. Various numerical
    formats can be specified using nform variable,
    including int (fixed point), fl32 (float32), 
    fl16 (float16), and bfl16 (bfloat16)
    """
    nform='fl16'  #numerical format for internal computations
    eps=1.e-30  #small number to avoid dividing by zero 
    
    #compute exp(r) and k, where exp(x)=2^k * exp(r)
    ey,k = poly2exp(x,nform=nform,dim=dim)
    kmax,_ = torch.max(k,dim=dim, keepdim=True)

    #compute sum with normalization for numerical stability
    ey = ey * 2**(k-kmax)
    sum_exp = torch.sum(ey, dim=dim, keepdim=True)
    
    #compute softmax
    y = ey/(sum_exp + eps)

    return y

def poly2exp(x, nform, dim=-1):
    r"""
    This function computes exp(x) using a range reduction
    technique x = r + k*log(2), where k is an integer and 
    -log(2)/2 < r < +log(2)/2.  exp(x) = 2^k * exp(r),
    where exp(r) is approximated by a second degree polynomial
    """
    ln2 = 0.69315  # log(2)
    invln2 = 1.4427  # 1 / log(2)
    #polynomial coefficients
    c0f=1.0
    c1f=1.015082
    c2f=0.503765  

    if nform == "int":
        scale = 14  # emulate integer arithmetic with 14 bit fractional part
        c0int = 2**scale  # poly coefficient c0 = 1
        c1int = round(1.0151 * 2**scale)  # poly coefficient c1
        # note poly coefficient c2 is 0.5 

        # range reduction to range -log(2)/2 < r < log(2)/2
        k = torch.round(x * invln2)
        r = x - k * ln2

        # compute exp(r) emulating fixed point arithmetic 
        rint = torch.round(r * 2**scale)
        
        mult_add1 = torch.round(c1int + 0.5*rint)
        mult_add2 = torch.round(c0int + torch.round(mult_add1*rint*2**(-scale)))
        ey = mult_add2 * 2**(-scale) #convert back to decimal number
        
    elif nform == "fl32":   
        ln2 = torch.tensor(ln2,dtype=torch.float32)
        invln2= torch.tensor(invln2,dtype=torch.float32)
        c0=torch.tensor(c0f,dtype=torch.float32)
        c1=torch.tensor(c1f,dtype=torch.float32)
        c2=torch.tensor(c2f,dtype=torch.float32)
        xfp32=x.float()
        # range reduction to range -log(2)/2 < r < log(2)/2
        k = torch.round(xfp32 * invln2)
        r = xfp32 - k * ln2
        
        # compute exp(r) in FL32
        mult_add1 = c1 +c2*r
        ey = c0 + mult_add1*r
        
    elif nform == "fl16":   
        ln2 = torch.tensor(ln2,dtype=torch.float16)
        invln2= torch.tensor(invln2,dtype=torch.float16)
        c0=torch.tensor(c0f,dtype=torch.float16)
        c1=torch.tensor(c1f,dtype=torch.float16)
        c2=torch.tensor(c2f,dtype=torch.float16)
        
        xfp16=x.half()
        # range reduction to range -log(2)/2 < r < log(2)/2
        xtmp=xfp16 * invln2
        k = torch.round(xtmp.float())
        k = k.half()
        r = xfp16 - k * ln2
        
        # compute exp(r) in FL16
        mult_add1 = c1 +c2*r
        ey = c0 + mult_add1*r  
        
    elif nform == "bfl16":   
        ln2 = torch.tensor(ln2,dtype=torch.bfloat16)
        invln2= torch.tensor(invln2,dtype=torch.bfloat16)
        c0=torch.tensor(c0f,dtype=torch.bfloat16)
        c1=torch.tensor(c1f,dtype=torch.bfloat16)
        c2=torch.tensor(c2f,dtype=torch.bfloat16)
        
        xfp16=x.bfloat16()
        # range reduction to range -log(2)/2 < r < log(2)/2
        xtmp=xfp16 * invln2
        k = torch.round(xtmp.float())
        k = k.bfloat16()
        r = xfp16 - k * ln2
        
        # compute exp(r) in BFL32
        mult_add1 = c1 +c2*r
        ey = c0 + mult_add1*r   
        
    else:   
        print("error: unsuported numerical format; reverting to float32")
        ln2 = torch.tensor(ln2,dtype=torch.float32)
        invln2= torch.tensor(invln2,dtype=torch.float32)
        c0=torch.tensor(c0f,dtype=torch.float32)
        c1=torch.tensor(c1f,dtype=torch.float32)
        c2=torch.tensor(c2f,dtype=torch.float32)
        xfp32=x.float()
        # range reduction to range -log(2)/2 < r < log(2)/2
        k = torch.round(xfp32 * invln2)
        r = xfp32 - k * ln2
        
        # compute exp(r) in FL32
        mult_add1 = c1 +c2*r
        ey = c0 + mult_add1*r
        
    ey = ey.float()
    k = k.float()          
    return ey, k

def base2softmax(x, dim=-1, **kwargs):
    r"""
    This function computes Softmax using base2exp
    function for the exp(x). Various numerical
    formats can be specified using nform variable,
    including int (fixed point), fl32 (float32), 
    fl16 (float16), and bfl16 (bfloat16)
    """
    nform='fl16'  #numerical format for internal computations
    eps=1.e-30  #small number to avoid dividing by zero 
    #compute exp(x) for input vector x
    #including integer vector k for normalization
    ey,k = base2exp(x,nform=nform,dim=dim)

    kmax,_ = torch.max(k,dim=dim, keepdim=True)

    #compute sum with normalization for numerical stability
    ey = ey*2**(k-kmax)  
    sum_ey = torch.sum(ey, dim=dim, keepdim=True)

    #compute softmax 
    y=ey/(sum_ey + eps)
    return y


def base2exp(x, nform, dim=-1):
    r"""
    This function computes exp(x) using a range reduction
    technique exp(x)=(2^k)*2^v, where k is an integer and 0<v<1
    2^v is approximated by a simple linear interpolation d+v.
    nform specifies the numerical format.
    """

    log2e_fp=1.4426950408889634; #log2(e)
    d=0.957 #minmax solution for d over the input range 0<v<1  
    
    if nform == "int":
        scale=14 #assuming 14 bits after binary fixed point 
        log2e=round(log2e_fp*2**scale)/2**scale #log2(e) 
        d=round(d*2**scale)/2**scale
        
        #range reduction to k and v
        z=x*log2e
        k=torch.floor(z)
        v=z-k
        v=torch.round(v*2**scale)/2**scale
    elif nform == "fl32":
        log2e = torch.tensor(log2e_fp,dtype=torch.float32)
        d= torch.tensor(d,dtype=torch.float32)
        
        #range reduction to k and v
        xfp32=x.float()
        z=xfp32*log2e
        k=torch.floor(z)
        v=z-k
    elif nform == "fl16":
        log2e = torch.tensor(log2e_fp,dtype=torch.float16)
        d= torch.tensor(d,dtype=torch.float16)
        #range reduction to k and v
        xfp16=x.half()
        z=xfp16*log2e
        k=torch.floor(z.float()) #floor not implemented for float16
        k=k.half()
        v=z-k
    elif nform == "bfl16":
        log2e = torch.tensor(log2e_fp,dtype=torch.bfloat16)
        d= torch.tensor(d,dtype=torch.bfloat16)
        
        #range reduction to k and v
        xfp16=x.bfloat16()
        z=xfp16*log2e
        k=torch.floor(z.float()) #floor not implemented for float16
        k=k.bfloat16()
        v=z-k
    else:
        print("error: unsuported numerical format; reverting to float32")
         
        log2e = torch.tensor(log2e_fp,dtype=torch.float32)
        d= torch.tensor(d,dtype=torch.float32)
        
        #range reduction to k and v
        xfp32=x.float()
        z=xfp32*log2e
        k=torch.floor(z)
        v=z-k
        
    #compute exp(v)
    two_pow_v = v + d

    ey = two_pow_v.float()
    k = k.float()
    return ey, k
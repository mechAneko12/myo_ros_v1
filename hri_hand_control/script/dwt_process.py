import numpy as np
import copy
import sys

#import only_dwt
import time

sq3 = np.sqrt(3); fsq2 = 4.0*np.sqrt(2);  #N = 32      #N = 2^n
c0 = (1 + sq3)/fsq2;    c1 = (3 + sq3)/fsq2             #Daubechies 4 coeff
c2 = (3 - sq3)/fsq2;    c3 = (1 - sq3)/fsq2

class DWT_N:
    def __init__(self, n):
        if n < 4:
            print("n must be (n>=4).")
            sys.exit(1)
        self.n = n
        self.fil = None

    def filter(self):
        nend = 4
        nd = copy.copy(self.n)
        fil = []
        while nd >= nend:
            _fil = np.zeros((nd,nd))
            nd //= 2
            for ind in range(nd):
                _fil[ind*2][ind*2] = c0
                _fil[ind*2][ind*2+1] = c1
                _fil[ind*2+1][ind*2] = c3
                _fil[ind*2+1][ind*2+1] = -c2
                if ind == nd -1:
                    _fil[ind*2][0] = c2
                    _fil[ind*2][1] = c3
                    _fil[ind*2+1][0] = c1
                    _fil[ind*2+1][1] = -c0
                else:
                    _fil[ind*2][ind*2+2] = c2
                    _fil[ind*2][ind*2+3] = c3
                    _fil[ind*2+1][ind*2+2] = c1
                    _fil[ind*2+1][ind*2+3] = -c0
            fil.append(_fil)
        self.fil = fil
        return fil
    
    @staticmethod
    def daube4(f, nd, fil):
        f_tmp = f[:nd]
        f_tmp = np.dot(fil, f_tmp)
        nd_2 = nd//2
        for ind, tmp in enumerate(f_tmp):
            if ind % 2 == 0:
                f[ind//2] = tmp
            else:
                f[ind//2+nd_2] = tmp

    def main(self, f):
        nend = 4
        nd = copy.copy(self.n)
        i = 0
        while nd >= nend:
            self.daube4(f, nd, self.fil[i])
            nd //= 2
            i += 1
    
    

if __name__ == '__main__':
    f = np.arange(10,42).astype(np.float64)
    dwt = DWT_N(32)
    dwt.filter()
    start = time.time()
    dwt.main(f)
    print(time.time()-start)
    print(f)
    """
    f_2 = np.arange(10,42).astype(np.float64)
    start = time.time()
    only_dwt.pyram(0, f_2, 32,1)
    print(time.time()-start)
    print(f_2)
    
    for a,b in zip(f,f_2):
        print(a-b)"""
# -*- coding: utf-8 -*-
import sys, os
from chainer import cuda, gradient_check, Variable
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from wavenet import Padding1d

xp = cuda.cupy
x = xp.random.uniform(-1.0, 1.0, (2, 2, 1, 2)).astype(xp.float32)
pad = 3
xv = Variable(x)
out = Padding1d(pad=pad)(xv)
print out.data[0,0]
y_grad = xp.ones((2, 2, 1, 2 + pad)).astype(xp.float32)
gradient_check.check_backward(Padding1d(pad=pad), (x,), y_grad, eps=1e-2)
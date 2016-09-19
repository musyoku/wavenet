# -*- coding: utf-8 -*-
import sys, os
from chainer import cuda, gradient_check, Variable
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from wavenet import Slice1d

xp = cuda.cupy
x = xp.random.uniform(-1.0, 1.0, (2, 2, 1, 5)).astype(xp.float32)
cut = 3
xv = Variable(x)
out = Slice1d(cut=cut)(xv)
print x[0,0]
print out.data[0,0]
y_grad = xp.ones((2, 2, 1, 5 - cut)).astype(xp.float32)
gradient_check.check_backward(Slice1d(cut=cut), (x,), y_grad, eps=1e-2)
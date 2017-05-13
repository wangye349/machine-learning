import scipy.optimize as opt
import numpy as np
import sys

points = []
def f(p):
    x = p
    z = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    points.append((x[0],x[1],z))
    return z
def fprime(p):
    x = p
    dx = -2 + 2*x[0] - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*x[1] - 200*x[0]**2
    return np.array([dx, dy])

init_point =(-2,-2)

result = opt.fmin_cg(f, init_point,fprime)
print result
# fmin_func = opt.__dict__[method]
# if method in ["fmin", "fmin_powell"]:
#     result = fmin_func(f, init_point)
# elif method in ["fmin_cg", "fmin_bfgs", "fmin_l_bfgs_b", "fmin_tnc"]:
#     result = fmin_func(f, init_point, fprime)
# elif method in ["fmin_cobyla"]:
#     result = fmin_func(f, init_point, [])
# else:
#     print "fmin function not found"
#     sys.exit(0)

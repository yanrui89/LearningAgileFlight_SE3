import casadi
import numpy as np
import math

# M = casadi.SX([[3,0,0],[0,3, 0],[0,0,3]])
# a = casadi.SX([2.0])
# print(casadi.exp(M))
# print(casadi.inv(M))
#x = casadi.SX.sym('x',2)
x = casadi.SX.sym('x',2,2)
y = casadi.SX.sym('y', 2, 1)
# y = casadi.SX([2.0, 3.0])
z = x@y
f = casadi.Function('f',[x, y],\
      [z],\
      ['x','y'],['r'])

r1 = casadi.jacobian(z, y)
# r2 = casadi.jacobian(f, y)
f1 = casadi.Function('f',[x, y],\
      [r1],\
      ['x','y'],['r1'])
r1 = f1(casadi.SX([[1,2],[3,4]]), casadi.SX([2.0, 3.0]))

# r1 = casadi.jacobian(r0, y)
# print(casadi.jacobian(r0, y))
print(r1)
# print(r2)

for i in np.arange(-np.pi,np.pi,0.01):
    xx = np.sin(i)
    yy = casadi.sin(i)
    if xx - yy != 0:
        print("here")







import mygrad as mg
import numpy as np


# x = mg.tensor(3.0)
# y = np.square(x)

# y.backward()

# aa = mg.tensor(1.0)
# a = np.array([[aa,2],[3,4]])

# b = np.array([3,4])

# c = np.matmul(a,b)

# print(c)

# c.backward()

import torch
import math
import numpy

def Rd2Rp(tra_ang):
    theta = 2*math.atan(magni(tra_ang))
    vector = norm(tra_ang+np.array([1e-8,0,0]))
    return [theta,vector]

def Rd2Rp_torch(tra_ang):
    theta = 2*torch.atan(magni_torch(tra_ang))
    vector = norm_torch(tra_ang+torch.tensor([1e-8,0,0],dtype=torch.float64))
    return [theta,vector]

def magni(vector):
    return np.sqrt(np.dot(np.array(vector),np.array(vector)))

def magni_torch(vector):
    return torch.sqrt(torch.dot(vector,vector))

## return the unit vector of a vector
def norm(vector):
    return np.array(vector)/magni(np.array(vector))

def norm_torch(vector):
    return vector/magni_torch(vector)

def toQuaternion(angle, dir):
    if type(dir) == list:
        dir = numpy.array(dir)
    dir = dir / numpy.linalg.norm(dir)
    quat = numpy.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat.tolist()

def toQuaternion_torch(angle, dir):

    dir = dir / torch.norm(dir)
    quat = torch.zeros(4, dtype=torch.float64)
    quat[0] = torch.cos(angle / 2)
    quat[1:] = torch.sin(angle / 2) * dir
    return quat

def dir_cosine(q): # world frame to body frame
      q = q / np.linalg.norm(q)
      C_B_I = np.vstack((
      np.hstack((1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2]))),
      np.hstack((2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1]))),
      np.hstack((2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
      ))
      return C_B_I
    


def dir_cosine_torch(q): # world frame to body frame
      q = q / torch.norm(q)
      C_B_I = torch.vstack((
      torch.hstack((1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2]))),
      torch.hstack((2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1]))),
      torch.hstack((2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
      ))
      return C_B_I
    


# x = torch.ones(5)  # input tensor
# y = torch.zeros(3)  # expected output
# w = torch.randn(5, 3, requires_grad=True)
# b = torch.randn(3, requires_grad=True)
# z = torch.matmul(x, w)+b
# zz = torch.sum(z)
# zz.backward()
# w.grad
# b.grad
# print("complete")

x = torch.tensor([-0.00862278,  0.00384869,  0.00742577], dtype = torch.float64)
x.requires_grad_()
az =torch.tensor([-0.00862278,  0.00384869,  0.00742577,1.0000], dtype = torch.float64)
y = Rd2Rp_torch(x)
z = toQuaternion_torch(y[0],y[1])
a = dir_cosine_torch(z)
b = dir_cosine_torch(az).detach()

e = torch.eye(3)
f = e - torch.matmul(a, b)
g = torch.trace(f)
g.backward()

x.grad()


xx = np.array([-0.00862278,  0.00384869,  0.00742577])
yy = Rd2Rp(xx)
zz = toQuaternion(yy[0],yy[1])
aa = dir_cosine(zz)

print(x.requires_grad)

print(x.requires_grad)
b = Rd2Rp(x)

print("complete")


#array([-0.71780025,  0.32038354,  0.61815628])


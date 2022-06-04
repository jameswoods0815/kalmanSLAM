import numpy as np


# get the skew matrix for (w1,w2,w3) R=exp(x) for the rotation matrix;
def getSkewMatrix(x):
    x=x.reshape(-1)
    if x.size==3:
        tmp=np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        return tmp
    if x.size==6:
        tmp1=np.block([[getSkewMatrix(x[3:]), x[:3].reshape(-1, 1)], [np.array([0, 0, 0, 0])]])
        return tmp1

#get the skew  matrix exp(x)=T
def getSkew_6x6 (x):
    x = x.reshape(-1)
    return np.block([[getSkewMatrix(x[3:]), getSkewMatrix(x[:3])], [np.zeros((3, 3)), getSkewMatrix(x[3:])]])


def circleDot(x):
    return np.block([[np.eye(3), -getSkewMatrix(x[:3])], [np.zeros((1, 6))]])

def pi_function(q):
    return q / q[2, :]


def diff_func_pi(q):
    q = q.reshape(-1)
    return (1 / q[2]) * np.array([[1, 0, -q[0] / q[2], 0], [0, 1, -q[1] / q[2], 0], [0, 0, 0, 0],
                                  [0, 0, -q[3] / q[2], 1]])

from operator import mul


import numpy as np


class ConvLayerGrad(object):

    """
    ConvLayer parameter object.
    This then is passed as an argument to all the convolution operations.

    N: Number of images in mini-batch
    C: Number of input feature maps
    K: Number of output feature maps

    D: Depth  of input image
    H: Height of input image
    W: Width  of input image

    T: Depth  of filter kernel
    R: Height of filter kernel
    S: Width  of filter kernel

    padding: amount of zero-padding around the given edge
    strides: factor to step the filters by in a given direction
    """

    def __init__(self, lib, dtype,
                 N, C, K,
                 D=1, H=1, W=1,
                 T=1, R=1, S=1,
                 pad_d=0, pad_h=0, pad_w=0,
                 str_d=1, str_h=1, str_w=1):

        # Compute the output spatial dimensions
        M = D - (T - 1) * str_d + 2 * pad_d
        P = H - (R - 1) * str_h + 2 * pad_h
        Q = W - (S - 1) * str_w + 2 * pad_w

        self.C = C
        self.K = K
        self.M = M
        self.P = P
        self.Q = Q
        self.NCK = (N, C, K)
        self.TRS = (T, R, S)
        self.DHW = (D, H, W)
        self.MPQ = (M, P, Q)
        self.padding = (pad_d, pad_h, pad_w)
        self.strides = (str_d, str_h, str_w)

        self.dimI = (C, D, H, W, N)
        self.dimF = (K, T, R, S, N)
        self.dimO = (C, M, P, Q, K)
        self.dimI2 = (C * D * H * W, N)
        self.dimF2 = (K * T * R * S, N)
        self.dimO2 = (C * M * P * Q, K)
        self.sizeI = reduce(mul, self.dimI, 1)
        self.sizeF = reduce(mul, self.dimF, 1)
        self.sizeO = reduce(mul, self.dimO, 1)
        self.nOut = reduce(mul, self.MPQ, 1) * C

    def fprop_slice(self, q, S, X, padding, strides):
        qs = q - padding
        sliceF = []
        sliceI = []
        for s in range(S):
            x = qs + s * strides
            if x >= 0 and x < X:
                sliceF.append(s)
                sliceI.append(x)
        return sliceF, sliceI



def fprop_conv_grad(layer, I, F, O, alpha=1.0, relu=False):
    """
    Forward propagate the inputs of a convolutional network layer to
    produce output

    Arguments:
        layer: the conv layer as a parameter object
        I (CPUTensor): inputs
        F (CPUTensor): the weights (filters)
        O (CPUTensor): outputs
        alpha (float): linear scaling
        relu (boolean): apply ReLu or not before output
                        (currently not implemented)
    """
    assert layer.sizeI == I.size
    assert layer.sizeF == F.size
    assert layer.sizeO == O.size

    M, P, Q = layer.MPQ
    C, D, H, W, N = layer.dimI
    K, T, R, S, N_F = layer.dimF
    C_O, M, P, Q, K_O = layer.dimO

    assert N_F == N
    assert K_O == K
    assert C_O == C

    pad_d, pad_h, pad_w = layer.padding
    str_d, str_h, str_w = layer.strides

    array_I = I.get().reshape(layer.dimI)
    array_F = F.get().reshape(layer.dimF)
    array_O = O.get().reshape(layer.dimO)

    for m in range(M):
        sliceT, sliceD = layer.fprop_slice(m, T, D, pad_d, str_d)

        for p in range(P):
            sliceR, sliceH = layer.fprop_slice(p, R, H, pad_h, str_h)

            for q in range(Q):
                sliceS, sliceW = layer.fprop_slice(q, S, W, pad_w, str_w)

                sliceTRS = np.array([
                    t * R * S + r * S + s
                    for t in sliceT
                    for r in sliceR
                    for s in sliceS], dtype=np.intp)

                sliceDHW = np.array([
                    d * H * W + y * W + w
                    for d in sliceD
                    for y in sliceH
                    for w in sliceW], dtype=np.intp)

                slicedF = array_F.reshape(
                    (K, -1, N))[:, sliceTRS, :].reshape((K, -1))
                slicedI = array_I.reshape(
                    (C, -1, N))[:, sliceDHW, :].reshape((C, -1))

                array_O[:, m, p, q, :] = alpha * \
                    np.dot(slicedI,  slicedF.T)
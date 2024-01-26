import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random
from scipy.optimize import minimize

# important changes:
# set_to_none = False
# if p.grad is None or p.grad.sum() == 0:

class CAGrad():
    def __init__(self, optimizer, c, rescale):
        self._optim = optimizer
        self.c = c
        self.rescale = rescale
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=False)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        ca_grad = self._conflict_averse(grads, has_grads)
        ca_grad = self._unflatten_grad(ca_grad, shapes[0])
        self._set_grad(ca_grad)
        return

    def _conflict_averse(self, grads, has_grads, shapes=None):

        G = torch.stack(grads)
        GG = torch.matmul(G, G.T).cpu()
        g0_norm = (GG.mean() + 1e-8).sqrt()

        x_start = np.ones(len(grads)) / len(grads)
        bnds = tuple((0, 1) for _ in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (self.c * g0_norm + 1e-8).item()
        def objfn(x):
            return (x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1)) + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x.astype(np.float32)
        ww = torch.tensor(w_cpu).to(grads[0].device)
        gw = torch.matmul(ww.unsqueeze(0), G).squeeze(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = sum(grads) / len(grads) + lmbda * gw

        if self.rescale == 0:
            new_grads = g
        elif self.rescale == 1:
            new_grads = g / (1 + self.c ** 2)
        elif self.rescale == 2:
            new_grads = g / (1 + self.c)
        else:
            raise ValueError('No support rescale type {}'.format(self.rescale))
        return new_grads

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=False)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: 
                #     continue
                # tackle the multi-head scenario
                # if p.grad is None:
                if p.grad is None or p.grad.sum() == 0:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

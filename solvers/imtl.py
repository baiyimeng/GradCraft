import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


# important changes:
# set_to_none = False
# if p.grad is None or p.grad.sum() == 0:

class IMTL():
    def __init__(self, optimizer):
        self._optim = optimizer
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
        imtl_grad = self._impartial_mtl(grads, has_grads)
        imtl_grad = self._unflatten_grad(imtl_grad, shapes[0])
        self._set_grad(imtl_grad)
        return

    def _impartial_mtl(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        imtl_grads, num_task = copy.deepcopy(grads), len(grads)

        grads_shared = [g[shared] for g in grads]
        norm_grads_shared = [g / torch.norm(g) for g in grads_shared]

        D = grads_shared[0] - torch.stack(grads_shared[1:]) # (T-1, n)
        U = norm_grads_shared[0] - torch.stack(norm_grads_shared[1:]) # (T-1, n)
        first_element = torch.matmul(grads_shared[0].unsqueeze(0), U.t()) # (1, T-1)
        try:
            second_element = torch.inverse(torch.matmul(D, U.t())) # (T-1, T-1)
        except:
            # workaround for cases where matrix is singular
            second_element = torch.inverse(torch.eye(num_task - 1, device=grads[0].device) * 1e-8 + torch.matmul(D, U.t()))
        alpha_ = torch.matmul(first_element, second_element).squeeze(0) # (1, T-1)
        alpha = torch.cat((1 - alpha_.sum().unsqueeze(0), alpha_))

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = torch.matmul(alpha.unsqueeze(0), torch.stack([g[shared] for g in imtl_grads])).squeeze(0)
        merged_grad[~shared] = torch.stack([g[~shared] for g in imtl_grads]).sum(dim=0)
        return merged_grad

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

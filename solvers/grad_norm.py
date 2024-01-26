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

class GradNorm():
    def __init__(self, optimizer, num_task, lr_weights, alpha, device):
        self._optim = optimizer
        
        weights = torch.ones(num_task)
        weights = (weights / sum(weights)).to(device)
        weights = nn.Parameter(weights)
        self.weights = weights

        self._optim_weights = torch.optim.Adam([weights], lr=lr_weights)

        self.alpha = torch.tensor(alpha, device=device)
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
        gn_grad = self._grad_norm(grads, has_grads, objectives)
        gn_grad = self._unflatten_grad(gn_grad, shapes[0])
        self._set_grad(gn_grad)
        return

    def _grad_norm(self, grads, has_grads, objectives, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        gn_grads, num_task = copy.deepcopy(grads), len(grads)
        
        # update weights
        loss_detach = torch.stack([l.detach() for l in objectives])
        self._optim_weights.zero_grad()
        weights_copy = copy.deepcopy(self.weights.detach())
        gn_grads_norm = [torch.norm(g) for g in gn_grads]
        weighted_gn_grads_norm = torch.stack(gn_grads_norm) * self.weights
        weighted_gn_grads_norm_average = torch.mean(weighted_gn_grads_norm).detach()
        loss_ratio = loss_detach / 0.693147181
        relative_inverse = loss_ratio / torch.mean(loss_ratio)
        gn_loss_list = [torch.abs(weighted_gn_grads_norm[i] - weighted_gn_grads_norm_average * (relative_inverse[i] ** self.alpha)) for i in range(num_task)]
        gn_loss = sum(gn_loss_list) / num_task
        gn_loss.backward()
        self._optim_weights.step()
        self.weights.data = self.weights.data / sum(self.weights.data)

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = torch.matmul(weights_copy.unsqueeze(0), torch.stack([g[shared] for g in gn_grads])).squeeze(0)
        merged_grad[~shared] = torch.stack([g[~shared] for g in gn_grads]).sum(dim=0)
        
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

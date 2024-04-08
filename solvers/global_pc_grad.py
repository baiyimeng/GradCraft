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

class GlobalPCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
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
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    # def _project_conflicting(self, grads, has_grads, shapes=None):
    #     shared = torch.stack(has_grads).prod(0).bool()
    #     pc_grads, num_task = copy.deepcopy(grads), len(grads)

    #     for i in range(num_task):
    #         conflict_task_index = []
    #         for j in range(num_task):
    #             if torch.dot(grads[i], grads[j]) < 0:
    #                 conflict_task_index.append(j)
    #         if len(conflict_task_index) == 0:
    #             continue
    #         elif len(conflict_task_index) == 1:
    #             j = conflict_task_index[0]
    #             g_j = grads[j]
    #             pc_grads[i] -= torch.dot(pc_grads[i], g_j) * g_j / (g_j.norm()**2)
    #         else:
    #             conflict_grads = [grads[j] for j in conflict_task_index]
    #             G = torch.stack(conflict_grads)
    #             left = torch.matmul(G, G.T)
    #             right = -torch.matmul(G, grads[i].unsqueeze(1))

    #             conflict_weights = torch.matmul(torch.inverse(left), right)
    #             pc_grads[i] += torch.matmul(G.T, conflict_weights).squeeze(-1)

    #     merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    #     if self._reduction:
    #         merged_grad[shared] = torch.stack([g[shared]
    #                                        for g in pc_grads]).mean(dim=0)
    #     elif self._reduction == 'sum':
    #         merged_grad[shared] = torch.stack([g[shared]
    #                                        for g in pc_grads]).sum(dim=0)
    #     else: exit('invalid reduction method')

    #     merged_grad[~shared] = torch.stack([g[~shared]
    #                                         for g in pc_grads]).sum(dim=0)
    #     return merged_grad

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(has_grads).prod(0).bool()
        grads = torch.stack(grads)
        num_task = len(grads)
        
        def proj_one(g):
            inner_products = torch.sum(g * grads, dim = -1)
            negative_indices = torch.where(inner_products < 0.)[0]
            if len(negative_indices) == 0:
                return g
            else:
                G = torch.gather(input=grads, index=negative_indices.unsqueeze(1).expand(-1, grads.size(1)), dim=0)
                left = torch.matmul(G, G.T)
                right = -torch.matmul(G, g.unsqueeze(1))
                conflict_weights = torch.matmul(torch.inverse(left), right)
                res = torch.matmul(G.T, conflict_weights).squeeze(-1)
                return g + res
        
        pc_grads = torch.stack([proj_one(g) for g in grads])
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                            for g in pc_grads]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                            for g in pc_grads]).sum(dim=0)
        else: exit('invalid reduction method')
    
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grads]).sum(dim=0)
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

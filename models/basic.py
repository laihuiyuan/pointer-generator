# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn

class BasicModule(nn.Module):
    def __init__(self, init='uniform'):
        super(BasicModule, self).__init__()
        self.init = init

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if self.init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif self.init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif self.init == 'truncated_normal':
                    self.truncated_normal_(param, mean=0,std=stddev)

    def truncated_normal_(self, tensor, mean=0, std=1.):
        """
        Implemented by @ruotianluo
        See https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
        """
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor
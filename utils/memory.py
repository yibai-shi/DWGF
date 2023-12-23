import numpy as np
import torch
from tqdm import tqdm
import os
from utils.util import *


class MemoryBank(object):
    def __init__(self, args, n, dim):
        self.n = n
        self.dim = dim
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)

    def reset(self):
        self.ptr = 0

    def update(self, features, targets):
        b = features.size(0)

        assert (b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr + b].copy_(features.detach())
        self.targets[self.ptr:self.ptr + b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to(self.device)


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, generator, device):
    model.eval()
    memory_bank.reset()

    for batch in tqdm(loader, desc='filling memory bank'):

        batch = tuple(t.cuda(non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        X = {"input_ids": generator.random_token_replace(input_ids.cpu()).to(device), 
            "attention_mask": input_mask, "token_type_ids": segment_ids}
        _, _, feature = model(X, output_hidden_states=True)

        memory_bank.update(feature, label_ids)

    memory_bank.cuda()

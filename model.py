import torch
from torch import nn
from torch.nn.parameter import Parameter

from utils.util import *


class BertForModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, self.num_labels)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        outputs = self.backbone(**X, output_hidden_states=True)
        CLSEmbedding = torch.mean(outputs.hidden_states[-1][:, 1:], dim=1)
        CLSEmbedding = self.dropout(CLSEmbedding)

        logits = self.classifier(CLSEmbedding)

        return CLSEmbedding, logits

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, y)
        return output 


class BertForGraph(nn.Module): 
    def __init__(self, model_name, num_labels):
        super(BertForGraph, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.degree = 1
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, self.backbone.config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.backbone.config.hidden_size, 256)
        )
        self.cluster_layer = Parameter(torch.Tensor(self.num_labels, self.backbone.config.hidden_size))

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        outputs = self.backbone(**X, output_hidden_states=True)
        CLSEmbedding = torch.mean(outputs.hidden_states[-1][:, 1:], dim=1)

        q = 1.0 / (1.0 + torch.sum(torch.pow(CLSEmbedding.unsqueeze(1) - self.cluster_layer, 2), 2) / self.degree)
        q = q.pow((self.degree + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        contrast_feat = self.projection(CLSEmbedding)

        return CLSEmbedding, q, contrast_feat
    
    def loss_self(self, q):
        weight = q ** 2 / q.sum(0)
        p = (weight.t() / weight.sum(1)).t()
        loss = F.kl_div(q.log(), p, reduction='batchmean')

        return loss

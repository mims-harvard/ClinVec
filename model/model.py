import dgl 
import torch 
import pytorch_lightning as pl
import gc
import torchmetrics
import torch.nn as nn 
import torch.nn.functional as F


# set seeds
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

"""
    config_dict: in_feat, out_feat, head_size_1, head_size_2, num_heads_1, num_heads_2, dropout
"""
class mini_hgt(pl.LightningModule):
    def __init__(self, config_dict):
        super().__init__()

        self.num_layers = config_dict['num_layers']
        in_feat = config_dict['in_feat']
        head_size = config_dict['head_size']
        num_heads = config_dict['num_heads']
        dropout = config_dict['dropout']
        out_feat = config_dict['out_feat']

        self.convs = torch.nn.ModuleList()

        for i in range(self.num_layers):
            in_dim = in_feat if i == 0 else head_size * num_heads
            conv = dgl.nn.pytorch.conv.HGTConv(in_dim, head_size, num_heads, 4, 1, use_norm=True, dropout=dropout)
            self.convs.append(conv) # 7
        self.linear = torch.nn.Linear(head_size * num_heads, out_feat)
        self.relu = torch.nn.ReLU()

    def forward(self, blocks, x):
        for i in range(self.num_layers):
            b = blocks[i]
            x = self.convs[i](b, x, b.ndata['ntype']['_N'],  # torch.zeros_like(b.ndata['ntype']['_N'])
                              torch.zeros_like(b.edata['etype']))
            
            #x = self.convs[i](b, x, torch.zeros_like(b.ndata['ntype']['_N']),  # torch.zeros_like(b.ndata['ntype']['_N'])
            #                  torch.zeros_like(b.edata['etype']))
            
            # if not last layer
            if i < self.num_layers - 1:
                x = self.relu(x)

        x = self.linear(x)

        return x

"""
    Contrastive learning model

    config_dict: hgt_config
"""
class EdgePredModel(pl.LightningModule):
    def __init__(self, homo_hg, hgt_config):
        super().__init__()

        self.lr = hgt_config['lr']

        self.homo_hg = homo_hg
        
        self.het_gnn = mini_hgt(hgt_config)

        self.accuracy = torchmetrics.classification.Accuracy(task='binary')
        self.softmax = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # save embeddings
        self.h_dict = {}
                 
    def forward(self, _, input_nodes, blocks):

        x_graph = blocks[0].srcdata['feat']

        # returns dictionary of embeddings
        h = self.het_gnn(blocks, x_graph)

        for ind, i in zip(input_nodes.tolist(), range(0, h.shape[0])):
            self.h_dict[ind] = h[i]
            
        # cleanup
        torch.cuda.empty_cache()
        gc.collect()
                    
    def training_step(self, train_batch, _):
        
        edge_input_nodes, edge_graph, edge_blocks, edge_batch_samples = train_batch
        
        # edges
        self.forward(edge_graph, edge_input_nodes, edge_blocks)
        edge_loss_score, acc = self.edge_loss(edge_batch_samples)

        self.log('train_loss_edge', edge_loss_score, prog_bar=True, batch_size=len(edge_batch_samples))
        self.log('train_acc_edge', acc, prog_bar=True, batch_size=len(edge_batch_samples)) 
        
        # reset embeddings dict
        self.h_dict = {}
            
        return edge_loss_score

    def validation_step(self, val_batch, _):
        edge_input_nodes, edge_graph, edge_blocks, edge_batch_samples = val_batch
        
        # edges
        self.forward(edge_graph, edge_input_nodes, edge_blocks)
        edge_loss_score, acc = self.edge_loss(edge_batch_samples)

        self.log('val_loss_edge', edge_loss_score, prog_bar=True, batch_size=len(edge_batch_samples))
        self.log('val_acc_edge', acc, prog_bar=True, batch_size=len(edge_batch_samples)) 
        
        # reset embeddings dict
        self.h_dict = {}
            
        return edge_loss_score
        
    def edge_loss(self, batch_samples):
        similarity_list = []

        for s in batch_samples:
            d_anch = s[0]
            d_pos = s[1]
            neg_samples = s[2:]
            anch_embed = self.h_dict[d_anch[0].item()]
            pos_embed = self.h_dict[d_pos[0].item()]
            neg_embed_list = [self.h_dict[d_neg[0].item()] for d_neg in neg_samples]          
            similarities = self.compute_sim(anch_embed, pos_embed, neg_embed_list)
            similarity_list.append(similarities)
            
        loss, acc = self.compute_infonce(similarity_list)
        
        return loss, acc
    
    def compute_infonce(self, similarity_list):
        
        scores = torch.stack(similarity_list)
        last_value = scores.shape[1] - 1
        acc = self.infonce_accuracy(scores)
        scores = torch.flip(scores, dims=(1,)) # reverse order of scores
        targets = torch.tensor([last_value]*len(scores), device=self.device)
        loss = F.cross_entropy(scores, targets)
        
        return loss, acc
    
    def compute_sim(self, anch_embed, pos_embed, neg_embed_list):
        neg_embed = torch.stack(neg_embed_list)
        all_anch_embed = anch_embed.repeat((len(neg_embed_list),1))
        neg_sim = self.cos(all_anch_embed, neg_embed).squeeze(dim=0)
        similarities = neg_sim  
        pos_sim = self.cos(anch_embed.unsqueeze(dim=0), pos_embed)
        similarities = torch.cat([pos_sim, similarities])
        return similarities
    
    def infonce_accuracy(self, preds):
        targets = torch.tensor([1]*len(preds), device=self.device)
        last_value = preds.shape[1] - 1
        preds = torch.flip(preds, dims=(1,)) # reverse order of scores       
        preds = torch.argmax(preds, dim=1)
        preds = torch.where(preds == last_value, 1, 0)
        acc = self.accuracy(preds, targets)
        return acc
    
    def configure_optimizers(self):
        param_list = [{'params': self.het_gnn.parameters()}]
        optimizer = torch.optim.Adam(param_list, lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, verbose=True)
        return [optimizer], [lr_scheduler]
    
 

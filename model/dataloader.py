import torch 
import pytorch_lightning as pl
import random
import dgl

# NOTES: sample by og node type or it will be too easy
# filter super high degree nodes (like infections)

# set seeds
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

"""
    dl_config: batch_size, sampler, n_neg, idx
"""
class edge_pred_dataloader(pl.LightningDataModule):              
    def __init__(self, homo_hg, homo_hg_dict, rev_edge_dict):
        super().__init__()

        e_n_neg = homo_hg_dict['n_neg']
        e_batch_size = homo_hg_dict['batch_size']
        num_layers = homo_hg_dict['num_layers']
        n_e_sampler = homo_hg_dict['sampler_n']
        fanout_list = [n_e_sampler] * num_layers
        e_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout_list, prob='p')
        #e_sampler = dgl.dataloading.MultiLayerNeighborSampler([100, 10], prob='p')

        self.edge_dataloader = typed_edge_dataloader(homo_hg, e_sampler, e_batch_size, e_n_neg, rev_edge_dict)
        self.e_batch_size = e_batch_size
                
    def __len__(self):
        n = len(self.edge_dataloader.train_chunks)
        return n
        
    def train_dataloader(self):
        return self.edge_dataloader


class typed_edge_dataloader():
    def __init__(self, homo_hg, sampler, edge_batch, n_neg, rev_edge_dict):

        self.homo_hg = homo_hg
        self.n_negs = n_neg
        self.sampler = sampler 
        self.all_nodes = set(homo_hg.nodes().tolist())

        # save list of nodes per ntype
        self.all_node_ntypes = {}
        for i in torch.unique(homo_hg.ndata['ntype']):
            self.all_node_ntypes[i.item()] = []

        for nid, ntype in zip(homo_hg.nodes(), homo_hg.ndata['ntype']):
            self.all_node_ntypes[ntype.item()].append(nid.item())

        # ntype lookup
        self.ntype_lookup = {}
        for nid, ntype in zip(homo_hg.nodes(), homo_hg.ndata['ntype']):
            self.ntype_lookup[nid.item()] = ntype.item()
 
        # reverse edges for exclusion
        self.rev_id_dict = rev_edge_dict
        
        edge_prob_map = {
            0: 1.0, 1: 5.0
        }
        
        self.homo_hg.edata['p'] = torch.tensor([edge_prob_map[x] for x in self.homo_hg.edata['etype'].tolist()])

        # train over ALL edges for full model training
        self.train_eids = self.homo_hg.edges(form='eid').tolist()
        random.shuffle(self.train_eids)
    
        self.train_chunks = [self.train_eids[i:i+edge_batch] for i in range(0, len(self.train_eids), edge_batch)]
            
    def neg_sampler(self, id):
        
        if torch.is_tensor(id):
            id = id.item()   

        # id for src and dst node
        src_id = self.homo_hg.find_edges(id)[0][0].item()
        dst_id = self.homo_hg.find_edges(id)[1][0].item()

        # get type of dst id
        dst_type_ind = self.ntype_lookup[dst_id]

        # sample from all possible dst nodes 
        curr_dst = self.homo_hg.out_edges(src_id)[1].tolist()

        # match negs by ntype
        cand_negs = list(set(self.all_node_ntypes[dst_type_ind]) - set(curr_dst))
        #cand_negs = list(self.all_nodes - set(curr_dst))
        neg_dst_list = random.sample(cand_negs, self.n_negs)
        neg_dst_list = torch.tensor(neg_dst_list)

        return neg_dst_list    

    def __len__(self):
        n = len(self.train_chunks)
        return n
    
    def __getitem__(self, idx):
        
        eid_chunk_list = self.train_chunks[idx]
        sample_list = []
        
        # get eid (loops over eids)
        for sample_eid in eid_chunk_list:
            # get list of neg samples (eid)
            neg_list = self.neg_sampler(sample_eid)

            # id for src and dst node
            src_id = self.homo_hg.find_edges(sample_eid)[0][0].item()
            dst_id = self.homo_hg.find_edges(sample_eid)[1][0].item()

            d_anch = torch.tensor([src_id])
            d_pos = torch.tensor([dst_id])
            triplet = [d_anch, d_pos]
            
            for neg_id in neg_list:
                d_neg = torch.tensor([neg_id])
                triplet.append(d_neg)
            
            # add triplet for a given eid
            sample_list.append(triplet)

        # get node ids for computaiton graph
        
        # merge into one list
        all_subgraph_list = torch.unique(torch.cat([x for sublist in sample_list for x in sublist]))

        # set seeds for sampling
        seed_list = all_subgraph_list
                
        # remove edges in minibatch and reverse edges
        rev_eids = [self.rev_id_dict[x] for x in eid_chunk_list]

        # add current eids from minibatch to remove 
        remove_eids = torch.tensor(eid_chunk_list + rev_eids)
        g = dgl.remove_edges(self.homo_hg, remove_eids)
        g, input_nodes, blocks = self.sampler.sample_blocks(g, seed_list)
                    
        return g, input_nodes, blocks, sample_list
                                 
    def make_rev_id_dict(self):
        # get forward/backward edge ids
        forward_eid_list = []
        for eid, type_ind in zip(self.homo_hg.edges(form='eid'), self.homo_hg.edata['etype'].detach().tolist()):
            if type_ind in [0]:
                forward_eid_list.append(eid.item())

        backward_eid_list = []
        for eid, type_ind in zip(self.homo_hg.edges(form='eid'), self.homo_hg.edata['etype'].detach().tolist()):
            if type_ind in [1]:
                backward_eid_list.append(eid.item())

        rev_id_dict = {}
        for k, v in zip(forward_eid_list, backward_eid_list):
            rev_id_dict[k] = v

        for k, v in zip(backward_eid_list, forward_eid_list):
            rev_id_dict[k] = v
        return rev_id_dict
   

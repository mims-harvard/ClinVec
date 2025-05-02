import dgl 
import torch 
import pandas as pd 
import pickle 
import pytorch_lightning as pl
import umap
import gc
import random
from tqdm.notebook import tqdm

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from model import EdgePredModel
from dataloader import edge_pred_dataloader

# set seeds
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

def open_pickle(f):
    with open(f, 'rb') as fname:
        node_dict = pickle.load(fname)
    return node_dict

def to_pickle(node_dict, f):
    with open(f, 'wb') as fname:
        pickle.dump(node_dict, fname)

device = "cuda"


#
# Set up model params
#
print("Assembling graph")
node_df = pd.read_csv("/n/holylfs06/LABS/mzitnik_lab/Lab/ruthjohnson/kg_paper_revision/connected_node_v3_df.csv", sep='\t')
edge_df = pd.read_csv("/n/holylfs06/LABS/mzitnik_lab/Lab/ruthjohnson/kg_paper_revision/connected_edge_v3_df.csv", sep='\t')

u = torch.tensor(edge_df['node_index_x'].tolist())
v = torch.tensor(edge_df['node_index_y'].tolist())

g = dgl.graph((u,v))
graph_feature_df = pd.read_csv("/n/holylfs06/LABS/mzitnik_lab/Lab/ruthjohnson/kg_paper_revision/graph_feature_v3_df.csv")
g.ndata['feat'] = torch.tensor(graph_feature_df.values, dtype=torch.float32)

ntype_list = node_df['ntype'].unique()
ntype_dict = {}
ntype_index_dict = {}
i = 0
for t in ntype_list:
    ntype_dict[t] = i
    ntype_index_dict[i] = t
    i+=1 

ntype_index_dict = {}
ntype_index_dict['ATC'] = 0
ntype_index_dict['CPT'] = 1
ntype_index_dict['ICD9CM'] = 2
ntype_index_dict['ICD10CM'] = 2
ntype_index_dict['LNC'] = 1
ntype_index_dict['PHECODE'] = 2
ntype_index_dict['RXNORM'] = 0
ntype_index_dict['SNOMEDCT_US'] = 3
ntype_index_dict['UMLS_CUI'] = 3

etypes = edge_df['ntype_x'] + ':' + edge_df['ntype_y']
etype_list = etypes.unique()
etype_dict = {}
i = 0 
for t in etype_list:
    if t.split(':')[1] in ['ATC', 'PHECODE', 'CPT']:
        etype_dict[t] = 1
    else:
        etype_dict[t] = 0

node_df['ntype_index'] = node_df['ntype'].map(ntype_index_dict)
g.ndata['ntype'] = torch.tensor(node_df['ntype_index'].tolist(), dtype=torch.int32)
g.edata['etype'] = torch.tensor((edge_df['ntype_x'] + ':' + edge_df['ntype_y']).map(etype_dict), dtype=torch.int32)

rev_id_dict = open_pickle("/n/holylfs06/LABS/mzitnik_lab/Lab/ruthjohnson/kg_paper_revision/model/rev_edge_dict_v3.pkl")

#
# Set up trainer
# 
n_epochs=10

config = {
    'num_layers': 4,
    'n_neg': 50,
    'batch_size': 500,
    'sampler_n': 20, 
    'lr': 1e-4,
    'in_feat': 128,
    'out_feat': 128,
    'head_size': 512,
    'num_heads': 3, 
    'dropout': 0.0
}
device = "cuda"

wandb_logger = WandbLogger(project="kg_gnn_small_ntype")
checkpoint_callback = ModelCheckpoint()

trainer = pl.Trainer(max_epochs=n_epochs,
                        log_every_n_steps=1,
                        precision="bf16-mixed", #"bf16-mixed"
                        accelerator=device,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback]
                        )


data_module = edge_pred_dataloader(homo_hg=g, homo_hg_dict=config, rev_edge_dict=rev_id_dict)

model = EdgePredModel(g, hgt_config=config)

torch.set_float32_matmul_precision('medium')
#with torch.no_grad():
    #model = EdgePredModel(homo_hg=g, hgt_config=config)
    #model = EdgePredModel.load_from_checkpoint("/n/home01/ruthjohnson/ruthjohnson/kg_paper_revision/model/kg_gnn_small_ntype/fl3bhz34/checkpoints/epoch=4-step=28580.ckpt", 
    #                                        homo_hg=g, hgt_config=config)
    #model.load_state_dict(torch.load("trained_hgt.pt"))


# train!
trainer.fit(model=model, datamodule=data_module,
             ckpt_path="/n/home01/ruthjohnson/ruthjohnson/kg_paper_revision/model/kg_gnn_small_ntype/fl3bhz34/checkpoints/epoch=4-step=28580.ckpt")

#
# Save trained model and weights
#
torch.save(model.het_gnn.state_dict(), "hgt_small_ntype_full.pt")

# compute embeddings
model = model.eval()
model.cuda()
model.homo_hg = model.homo_hg.to(torch.device('cuda:0'))

#eval_list = node_df['node_index'].tolist()
name_degree_df = pd.read_csv("/n/holylfs06/LABS/mzitnik_lab/Lab/ruthjohnson/kg_paper_revision/name_degree_df.csv")
eval_list = name_degree_df.loc[name_degree_df['n'] < 1000]['node_index'].tolist()

with torch.no_grad():
    #all_node_ids = model.homo_hg.nodes()
    all_node_ids = torch.tensor(eval_list, device=model.device)
    
    model.eval() 

    sampler = dgl.dataloading.MultiLayerNeighborSampler([1000, 100, 100, 100])
    #sampler = dgl.dataloading.MultiLayerFullNeighborSampler(4)

    dataloader = dgl.dataloading.DataLoader(
        model.homo_hg,
        all_node_ids.cuda(),     # compute embeddings for all the nodes
        sampler,
        shuffle=False,    # remember to set this to False so you can just concatenate embeddings at the end
        batch_size=1,
        device="cuda"
    )

    h_list = []

    for inputs, graph, blocks in tqdm(dataloader):
        inputs = blocks[0].srcdata['feat']
        graph_h = model.het_gnn(blocks, inputs)
        h_list.append(graph_h)

        torch.cuda.empty_cache()
        gc.collect()

all_h = torch.concat(h_list).cpu()
to_pickle(all_h, "gnn_embeds_small_ntype_full_neighbor.pkl")
        
reducer = umap.UMAP(n_neighbors=5, min_dist=0.005, metric='cosine')
embedding_standard = reducer.fit_transform(all_h.numpy())
embed_df = pd.DataFrame(embedding_standard)
embed_df['node_index'] = eval_list
embed_df = embed_df.merge(node_df, on='node_index')
embed_df.to_csv("embed_small_ntype_full_neighbor_df.csv", index=False)


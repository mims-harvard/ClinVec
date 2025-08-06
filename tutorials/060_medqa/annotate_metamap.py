from mmlrestclient import *
from datasets import load_dataset
import pandas as pd
from tqdm.notebook import tqdm

ds = load_dataset("VodLM/medqa", "tw") # tw, us

node_df = pd.read_csv("/n/home01/ruthjohnson/kg_paper_revision/connected_node_v2_df.csv", sep='\t')
keep_cui = set(node_df.loc[node_df['ntype'] == 'UMLS_CUI']['node_id'].str.split(':', expand=True)[0].tolist())
url='https://ii.nlm.nih.gov/metamaplite/rest/annotate'
acceptfmt = 'text/plain'

qa_rows = []

for idx in tqdm(range(0, len(ds['train']))):
    if (idx % 100) == 0:
           print(idx)
    q=ds['train'][idx]['question']
    a0 = ds['train'][idx]['answers'][0]
    a1 = ds['train'][idx]['answers'][1]
    a2 = ds['train'][idx]['answers'][2]
    a3 = ds['train'][idx]['answers'][3]
    qa_list = []

    for qa_str in [q, a0, a1, a2, a3]:
            params = [('inputtext', qa_str),
                    ('docformat', 'freetext'),
                            ('resultformat', 'mmi'), 
                            ('sourceString', 'ICD9CM'), ('sourceString', 'ICD10CM'), ('sourceString', 'ATC'), ('sourceString', 'RXNORM'),
                            ('sourceString', 'LNC'), ('sourceString', 'SNOMEDCT_US'), ('sourceString', 'CPT'),
                            ('semanticTypeString', 'all')
                            ]

            resp = handle_request(url, acceptfmt, params)
            text_list = resp.text.split('\n')
            text_list = [x for x in text_list if x]
            if len(text_list) > 0:
                    cui_list = set([x.split('|')[4] for x in text_list if float(x.split('|')[2]) > 0])
                    cui_list = list(cui_list & keep_cui)
            else:
                    cui_list = []
            qa_list.append(cui_list)
    qa_rows.append(qa_list)
qa_cui_df = pd.DataFrame(qa_rows)
qa_cui_df = qa_cui_df.assign(idx=qa_cui_df.index)

qa_cui_df.to_csv("qa_cui_df.csv", sep='\t')
import os
import json
import copy
import torch
import logging
import transformers
from random import shuffle
from transformers import Trainer
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, Sequence, List
import torch.nn as nn 
from custom_model import CustomModel
from transformers import LlamaConfig
import pandas as pd 
from random import shuffle 

import pickle 

def to_pickle(df, f):
    with open(f, 'wb') as fname:
        pickle.dump(df, fname)

def open_pickle(f):
    with open(f, 'rb') as file:
        data = pickle.load(file)
    return data  

SEED = 42
transformers.set_seed(SEED)
IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    is_lora: Optional[bool] = field(default=False)
    lora_rank: Optional[int] = field(default=16)
    target_modules :Optional[List[str]] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    use_cache : bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    gradient_clipping : float = field(
        default=None
    )
    
    
def jsonl_load(data_path):
    """Load a .jsonl file into a dictionary."""
    filepaths = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if filename.endswith('.jsonl')]    
    src_dict_ls = []
    for filepath in filepaths:
        lang = os.path.basename(filepath).split(".")[0]
        with open(filepath, "r", encoding='utf-8') as f:
            for line in f:
                src_dict = json.loads(line)
                src_dict["lang"] = lang
                src_dict_ls.append(src_dict)
                        
    res_dict_ls = []
    for src_dict in src_dict_ls:
        lang = src_dict["lang"]
        question = src_dict["question"]
        options = ""
        for key in src_dict["options"].keys():
            content = src_dict["options"][key]
            options += f"{key}. {content} "
        if isinstance(src_dict["answer_idx"], str):
            answer_id = src_dict["answer_idx"]
        elif isinstance(src_dict["answer_idx"], list):
            answer_id = ",".join(src_dict["answer_idx"])

        rationale = src_dict["rationale"]
        data_with_rationale = {
            "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account in {lang}. Let’s solve this step-by-step. You should first give the reason in {lang} for your choice. Then you should give the right answer index of the question.",
            "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
            "output":f"###Rationale: {rationale}\n###Answer: OPTION {answer_id} IS CORRECT."
        }    
        res_dict_ls.append(data_with_rationale)
        
        data_without_rationale = {
            "instruction" : f"You're a {lang} doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
            "input":f"###Question: {question} Which of the following is the best treatment for this patient? ###Options: {options}",
            "output":f"###Answer: OPTION {answer_id} IS CORRECT."
        }    
        res_dict_ls.append(data_without_rationale) 
               
    # shuffle the data for training
    #shuffle(res_dict_ls)                
    return res_dict_ls

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


#
# THIS IS WHERE TOKENIZING HAPPENS
# need to add context tokens in id's, but add ignore index -100 for labels
# add tokens right before '###Options:'
#
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, new_list_data_dict=None):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        
        if new_list_data_dict is not None:
            logging.warning("Using precomputed data dict...")
            list_data_dict = list(new_list_data_dict) 
            shuffle(list_data_dict)
        else:
            list_data_dict = jsonl_load(data_path)
            shuffle(list_data_dict)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #print(dict(input_ids=self.input_ids[i], labels=self.labels[i]))
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        #print(instances)
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        #print(input_ids)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, new_list_data_dict=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, new_list_data_dict=new_list_data_dict)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    dir = training_args.output_dir
    if not os.path.exists(dir):
        os.makedirs(dir)

    cui_embeds = open_pickle("/n/home01/ruthjohnson/kg_paper_revision/model/umls_cui_embeds.pkl")
    #cui_embeds = cui_embeds[0:5, :]
    new_list_data_dict = open_pickle("new_list_data_dict.pkl")

    node_df = pd.read_csv("/n/home01/ruthjohnson/kg_paper_revision/connected_node_v2_df.csv", sep='\t')
    keep_cui = set(node_df.loc[node_df['ntype'] == 'UMLS_CUI']['node_id'].str.split(':', expand=True)[0].tolist())
    #keep_cui = list(keep_cui)[0:5]

    config = LlamaConfig.from_pretrained("Henrychur/MMed-Llama-3-8B-EnIns")
    config.architectures = ["CustomModel"]
    model = CustomModel(config=config).from_pretrained("Henrychur/MMed-Llama-3-8B-EnIns")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        use_fast=False,
        trust_remote_code=True
    )

    # add new tokens and resize
    cui_tokens = [('<%s>' % x) for x in list(keep_cui)]
    tokenizer.add_tokens(cui_tokens, special_tokens=True)
    #model.resize_token_embeddings(len(tokenizer))
    model.setup(cui_embeds)

    config = LoraConfig(
        r = model_args.lora_rank,
        lora_alpha = 32, #model_args.lora_alpha,
        target_modules = model_args.target_modules,
        lora_dropout = 0.05,
        bias = 'none',
        task_type="CAUSAL_LM",
    )

    print("Tokenizer length: ", len(tokenizer))
    
    model = get_peft_model(model, config)
    model.cuda()
    
    tokenizer.pad_token = tokenizer.eos_token
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_args,
                                            new_list_data_dict=new_list_data_dict)
    #training_args.max_steps = 100

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    torch.save(model.proj_layer.state_dict(), os.path.join(dir, 'model_weights.pth'))



if __name__ == "__main__":
    train()

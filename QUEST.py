import logging
import sys
from argparse import ArgumentParser
import pandas as pd
import os
from os.path import join
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_scheduler, LlamaTokenizer
from model import LlamaPromptTuningLM
from evaluation import evaluate
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def setup_arg_parser():
    parser = ArgumentParser('structural_contrastive_learning')
    
    # Model and data parameters
    ### The model can be adapted to other variants. The one below is the same model mentioned in the paper.
    parser.add_argument('--model', type=str, default='decapoda-research/llama-7b-hf')
    ### Select your own dataset directory
    parser.add_argument('--dataset', type=str, default='dataset/LF-WikiSeeAlso')
    parser.add_argument('--tokenizer-max-len', type=int, default=64)
    parser.add_argument('--number-prompt-token', type=int, default=20)
    
    # Training parameters
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=18)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', type=str, default='linear', choices=['linear', 'cosine'])
    parser.add_argument('--num-warmup-steps', type=int, default=0)
    parser.add_argument('--ConWeight', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=2)
    
    # Logging and evaluation
    parser.add_argument('--experiment', type=str, default='default')
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-only', action='store_true')
    
    return parser.parse_args()

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def val(self):
        return self.sum / self.count if self.count != 0 else 0

class RandIndexDataset(Dataset):
    def __init__(self, range):
        self.range = range

    def __len__(self):
        return self.range

    def __getitem__(self, idx):
        return idx

class PairLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, x0, x1):
        a_n, b_n = x0.norm(dim=1)[:, None], x1.norm(dim=1)[:, None]
        a_norm = x0 / torch.max(a_n, self.eps * torch.ones_like(a_n))
        b_norm = x1 / torch.max(b_n, self.eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1)) / self.temperature
        targets = torch.arange(x0.shape[0]).long().to(x0.device)
        return self.loss(sim_mt, targets)

def get_training_pairs(target_index, titles, contents, max_num, labels, batch_size):
    com_text = np.array(['The title is ' + titles[i] + '. The content is ' + contents[i] 
                        for i in range(len(titles))], dtype=object)
    training_pairs = []
    
    for i, tgt in enumerate(target_index):
        for j, ind in enumerate(tgt):
            if j >= max_num:
                break
            training_pairs.append((com_text[i], labels[ind]))

    training_pairs = np.array(training_pairs, dtype=object)
    dataset = RandIndexDataset(len(training_pairs))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return training_pairs, loader

def setup_logging(experiment_name):
    os.makedirs("out", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"out/{experiment_name}.log"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger('structural_contrastive_learning')
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)
    return log

def initialize_model(args):
    model = LlamaPromptTuningLM.from_pretrained(
        args.model,
        n_tokens=args.number_prompt_token,
        initialize_from_vocab=True,
        dev=args.device
    )
    model.config.output_hidden_states = True
    return model

def initialize_optimizer(model, args):
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight"],
            "weight_decay": args.weight_decay,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.epochs,
    )
    return optimizer, scheduler

def main():
    args = setup_arg_parser()
    log = setup_logging(args.experiment)
    log.info(args)
    
    device = torch.device(args.device)
    model = initialize_model(args)
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = "[PAD]"
    tokenizer.model_max_length = args.tokenizer_max_len
    
    optimizer, scheduler = initialize_optimizer(model, args)
    loss_func = PairLoss(temperature=args.temperature).to(device)
    loss_meter = AverageMeter()
    
    # Load data
    trn_js = pd.read_json(join(args.dataset, 'trn.json'), lines=True)
    tst_js = pd.read_json(join(args.dataset, 'tst.json'), lines=True)
    lbl_js = pd.read_json(join(args.dataset, 'lbl.json'), lines=True)
    tst_content = tst_js.content
    
    if args.eval_only:
        full_prompt = load_obj('The weight of soft prompt')
        model.load_state_dict(torch.load('The_trained_model.pt'))
        model.soft_prompt.weight = torch.nn.Parameter(full_prompt)
        model = model.bfloat16().to(device)
        prec = evaluate(
            tokenizer, model, tst_js.title, tst_content, lbl_js.title, 
            tst_js.target_ind, tst_js.uid, lbl_js.uid,
            batch_size=args.eval_batch_size, 
            device=device, 
            print_freq=args.print_freq,
        )
        return
    
    best_metric = 0
    for epoch in range(args.epochs):
        training_pairs, loader = get_training_pairs(
            trn_js.target_ind, trn_js.title, trn_js.content, 4, lbl_js.title, args.batch_size
        )
        log.info(f"New Training Set Size: {len(training_pairs)}")
        
        model.train()
        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            
            pair_idx = batch.cpu().numpy()
            sentence_pairs = training_pairs[pair_idx]
            s0, s1 = sentence_pairs[:,0], sentence_pairs[:,1]
            
            t0 = tokenizer(list(s0), padding=True, truncation=True, 
                          max_length=args.tokenizer_max_len, return_tensors="pt")
            t1 = tokenizer(list(s1), padding=True, truncation=True, 
                          max_length=args.tokenizer_max_len, return_tensors="pt")
            
            t0 = {k: v.to(device) for k, v in t0.items()}
            t1 = {k: v.to(device) for k, v in t1.items()}

            o0 = model(**t0).hidden_states[-1][:,-1,:].to(device)
            o1 = model(**t1).hidden_states[-1][:,-1,:].to(device)

            loss = loss_func(o0, o1)
            loss_meter.update(loss.item())

            if batch_idx % args.print_freq == 0:
                log.info(f'Epoch {epoch}/{args.epochs}, Iter {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')

            loss.backward()
            optimizer.step()

        loss_meter.reset()
        scheduler.step()

        prec = evaluate(
            tokenizer, model, tst_js.title, tst_content, lbl_js.title,
            tst_js.target_ind, tst_js.uid, lbl_js.uid,
            batch_size=args.eval_batch_size,
            device=device,
            print_freq=args.print_freq,
        )
        
        if prec > best_metric:
            log.info('Saving model...')
            best_metric = prec
            torch.save(model.state_dict(), f'models/{args.experiment}.pt')

if __name__ == "__main__":
    main()
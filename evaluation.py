import numpy as np
import torch
import faiss
from beir.retrieval.evaluation import EvaluateRetrieval

def top_k_metrics(index, queries, label_indices, inst_uids, label_uids, topk_list=[1, 3, 5]):
    topk = max(topk_list)
    D, I = index.search(queries, topk)
    
    qrels = {}
    results = {}
    for idx, targets in enumerate(label_indices):
        qrels[inst_uids[idx]] = {label_uids[t]: 1 for t in targets}
        results[inst_uids[idx]] = {label_uids[I[idx][k]]: float(D[idx][k]) for k in range(topk)}

    return EvaluateRetrieval.evaluate(qrels, results, topk_list)

def evaluate(tokenizer, model, titles, contents, labels, target_label_indices, inst_uids, label_uids, batch_size=1560, device=torch.device('cuda'), print_freq=100):
    torch.cuda.empty_cache()
    model.eval()
    
    # sent = np.array([titles[i] + '\t' + contents[i] for i in range(len(titles))], dtype=object)
    ### align title and content
    sent = np.array(['The title is ' + titles[i] + '. The content is ' + contents[i] for i in range(len(titles))], dtype=object)

    idx = 0
    n = len(titles)
    inst_emb = []
    print('Embedding instances...')
    with torch.no_grad():
        while idx < n:
            tokens = tokenizer(sent[np.arange(idx, min(idx + batch_size, n))].tolist(), padding=True, truncation=True, return_tensors="pt")
            for k in tokens:
                tokens[k] = tokens[k].to(device)
            # emb = model(**tokens).last_hidden_state[:, 0]
            # emb = model(**tokens).hidden_states[-1].mean(dim = 1)
            emb = model(**tokens).hidden_states[-1][:,-1,:]
            inst_emb.append(emb.cpu())
            
            if (idx // batch_size) % print_freq == 0:
                print(f'{idx}/{n}')

            idx += batch_size
 
    idx = 0
    n = len(labels)
    labels = np.array(labels, dtype=object)
    label_emb = []
    print('Embedding labels...')
    with torch.no_grad():
        while idx < n:
            tokens = tokenizer(labels[np.arange(idx, min(idx + batch_size, n))].tolist(), padding=True, truncation=True, return_tensors="pt")
            for k in tokens:
                tokens[k] = tokens[k].to(device)
            # emb = model(**tokens).last_hidden_state[:, 0]
            # emb = model(**tokens).hidden_states[-1].mean(dim = 1)
            emb = model(**tokens).hidden_states[-1][:,-1,:]
            label_emb.append(emb.cpu())
            
            if (idx // batch_size) % print_freq == 0:
                print(f'{idx}/{n}')

            idx += batch_size

    # inst_emb = torch.cat(inst_emb, dim=0).numpy()
    inst_emb = np.array(torch.cat(inst_emb, dim=0).to(torch.float32))
    # label_emb = torch.cat(label_emb, dim=0).numpy()
    label_emb = np.array(torch.cat(label_emb, dim=0).to(torch.float32))

    faiss.normalize_L2(inst_emb)
    faiss.normalize_L2(label_emb)

    index = faiss.IndexFlatIP(inst_emb.shape[1])
    index.add(label_emb)
 
    top_k = [1, 3, 5, 10, 100]
    metrics = top_k_metrics(index, inst_emb, target_label_indices, inst_uids, label_uids, topk_list=top_k)
    for m in metrics:
        print(m)

    return metrics[-1]['P@1']

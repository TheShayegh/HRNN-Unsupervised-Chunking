import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import BucketIterator
from library.logger import timing_logger
import transformers
import os
from subprocess import run, PIPE
# from transformers import *

BATCH_SIZE = 1

class HRNNtagger(nn.ModuleList):
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        tagset_size: int,
        device: torch.device,
        train_embeddings: bool = False,
        vocab_size: int = None,
    ) -> None:

        super(HRNNtagger, self).__init__()
        self.device = device
        self.criterion = nn.NLLLoss().to(device)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.train_embeddings = train_embeddings
        if train_embeddings:
            self.vocab_size = vocab_size
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=padding_idx).to(self.device)
        self.rnn11 = nn.RNNCell(self.embedding_dim, self.hidden_dim).to(self.device)
        self.rnn12 = nn.RNNCell(self.embedding_dim, self.hidden_dim).to(self.device)
        self.rnn21 = nn.RNNCell(self.hidden_dim, self.hidden_dim).to(self.device)
        self.hidden2tag = nn.Linear(self.hidden_dim+self.hidden_dim+self.embedding_dim, self.tagset_size).to(self.device)
        self.soft = nn.Softmax(dim=1).to(self.device)
        

    def forward(
        self,
        h_init: torch.Tensor,
        x: torch.Tensor,
        seqlens: int,
        mask_ind = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        output_seq = torch.zeros((seqlens, self.tagset_size)).to(self.device)

        h11 = h_init.to(self.device)
        h12 = h_init.to(self.device)
        h1_actual = h_init.to(self.device)

        h21 = h_init.to(self.device)
        h22 = h_init.to(self.device)
        h2_actual = h_init.to(self.device)

        embeddings = self.word_embeddings(x) if self.train_embeddings else x
        for t in range(seqlens):
            entry = torch.unsqueeze(embeddings[t], 0).to(self.device)
            next_entry = torch.unsqueeze(embeddings[t], 0).to(self.device) \
                        if t == seqlens-1 else \
                        torch.unsqueeze(embeddings[t+1], 0).to(self.device)
            h11 = self.rnn11(entry, h1_actual)
            h12 = self.rnn12(entry, h_init)
            h22 = h2_actual
            h21 = self.rnn21(h1_actual, h2_actual)

            if t == 0:
                h1_actual = mask_ind*h12 + (1-mask_ind)*h11
                h2_actual = mask_ind*h21 + (1-mask_ind)*h22
                h_init = h1_actual
            else:
                h1_actual = torch.mul(h11, output[0]) + torch.mul(h12, output[1])
                h2_actual = torch.mul(h22, output[0]) + torch.mul(h21, output[1])

            tag_rep = self.hidden2tag(torch.cat((h1_actual, h2_actual, next_entry), dim=1)).to(self.device)
            output = torch.squeeze(self.soft(tag_rep))
            output_seq[t] = output

        return output_seq, h2_actual


    def init_hidden(self):
        return (torch.zeros(BATCH_SIZE, self.hidden_dim))




def get_training_equipments(
	model: HRNNtagger,
    lr: float,
    num_iter: int,
    warmup: int,
) -> tuple[torch.optim.Optimizer, transformers.SchedulerType]:
    optimizer = optim.Adam(model.parameters(), lr=lr*(num_iter+1)/num_iter, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup,
        num_training_steps=num_iter+1,
        num_cycles=0.5,
        last_epoch=-1
    )
    scheduler.step()
    return optimizer, scheduler




def make_bucket_iterator(
    data,
    device: torch.device,
):
    bucket_iterator = BucketIterator(
        data, 
        batch_size=BATCH_SIZE,
        sort_key=lambda x: np.count_nonzero(x[0]),
        sort=False, 
        shuffle=False,
        sort_within_batch=False,
        device=device,
    )
    bucket_iterator.create_batches()
    return bucket_iterator




def _forward(
    model: HRNNtagger,
    batch: torch.Tensor,
    hc: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding = batch[0][0].to(device)
    tag = batch[0][1].to(device)
    seqlens = torch.as_tensor(torch.count_nonzero(tag, dim=-1), dtype=torch.int64, device='cpu')+2
    tag = (tag - 1)[1:seqlens-1]

    model.zero_grad()
    tag_scores,_ = model(hc, embedding, seqlens)
    tag_scores = torch.log(tag_scores[1:seqlens-1])

    selection = (tag != 2)

    loss = model.criterion(tag_scores[selection], tag[selection])
    return tag_scores, loss # predicted probs, true tags, loss




@timing_logger
def train(
    model: HRNNtagger,
    data,
    optimizer,
    scheduler,
    device,
) -> float:
    model.train()
    loss_sum = 0.
    bucket_iterator = make_bucket_iterator(data, device=device)
    for batch in tqdm(bucket_iterator.batches, total=len(bucket_iterator)):
        hc = model.init_hidden().to(device)
        _, loss = _forward(model, batch, hc, device)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    scheduler.step()
    return loss_sum / len(bucket_iterator)




def validation_output(
    ind,
    true_tag,
) -> str:
    output = "x y B B\n"
    for i, pred in enumerate(ind[:-1]):
        true_label = true_tag[i+1]
        if true_label not in ["B", "I"]:
            continue
        predicted_label = "B" if pred else "I"
        output += f"x y {true_label} {predicted_label}\n"
    return output




def enforced_validation_output(
    ind,
    true_tag,
    enforced_tags,
) -> str:
    output = ""
    if enforced_tags[0] in ["B", "I"]:
        output += f"x y B {enforced_tags[0]}\n"
    else:
        output += f"x y B B\n"
    for i, pred in enumerate(ind[:-1]):
        true_label = true_tag[i+1]
        if true_label not in ["B", "I"]:
            continue
        if enforced_tags[i+1] in ["B", "I"]:
            predicted_label = enforced_tags[i+1]
        else:
            predicted_label = "B" if pred else "I"
        output += f"x y {true_label} {predicted_label}\n"
    return output




def enforced_Bstarting_validation_output(
    ind,
    true_tag,
    enforced_tags,
) -> str:
    output = ""
    if enforced_tags[0] in ["B", "I"]:
        output += f"x y B {enforced_tags[0]}\n"
    else:
        output += f"x y B B\n"
    for i, pred in enumerate(ind[:-1]):
        true_label = true_tag[i+1]
        if true_label not in ["B", "I"]:
            continue
        if enforced_tags[i+1] in ["B", "I"]:
            predicted_label = enforced_tags[i+1]
        elif enforced_tags[i] in ["B", "I"]:
            predicted_label = "B"
        else:
            predicted_label = "B" if pred else "I"
        output += f"x y {true_label} {predicted_label}\n"
    return output




@timing_logger
def validate(
    model: HRNNtagger,
    data,
    true_tags,
    device,
    enforced_mode: str = 'normal',
    enforced_tags = None,
) -> tuple[float, str]:
    model.eval()
    hc = model.init_hidden().to(device)
    loss_sum = 0.
    bucket_iterator = make_bucket_iterator(data, device=device)
    output = ""
    with torch.no_grad():
        for i, (batch, true_tag) in tqdm(enumerate(zip(bucket_iterator.batches, true_tags)), total=len(bucket_iterator)):
            tag_scores, loss = _forward(model, batch, hc, device)
            loss_sum += loss.item()
            ind = torch.argmax(tag_scores, dim=1)
            if enforced_tags is None:
                output += validation_output(ind, true_tag)
            else:
                enforced_tag = enforced_tags[i]
                if enforced_mode == 'bstarting':
                    output += enforced_Bstarting_validation_output(ind, true_tag, enforced_tag)
                else:
                    output += enforced_validation_output(ind, true_tag, enforced_tag)
    return loss_sum / len(bucket_iterator), output




def eval_conll2000(
    pairs: str,
    eval_conll_path: str = 'library/eval_conll.pl',
) -> tuple[float, float]: # F1, Acc
    pipe = run(["perl", eval_conll_path], stdout=PIPE, input=pairs, encoding='ascii')
    output = pipe.stdout.split('\n')[1]
    tag_acc = float(output.split()[1].split('%')[0])
    phrase_f1 = float(output.split()[-1])
    return phrase_f1, tag_acc
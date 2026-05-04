import torch
import torch.nn as nn
import torch.nn.functional as F
from models import FPN_OPS, HEAD_OPS, AGG_OPS, N_FPN_BLOCKS, N_HEAD_OPS

MAX_POOL_SIZE = 3 + N_FPN_BLOCKS

class NASController(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.input_size = max(MAX_POOL_SIZE, len(FPN_OPS), len(AGG_OPS), len(HEAD_OPS))
        self.hidden = hidden
        self.lstm = nn.LSTMCell(self.input_size, hidden)

        self.dec_id = nn.Linear(hidden, MAX_POOL_SIZE)
        self.dec_op = nn.Linear(hidden, len(FPN_OPS))
        self.dec_agg = nn.Linear(hidden, len(AGG_OPS))
        self.dec_hop = nn.Linear(hidden, len(HEAD_OPS))
        self.dec_share = nn.Linear(hidden, N_HEAD_OPS + 1)

        self.h0 = nn.Parameter(torch.zeros(1, hidden))
        self.c0 = nn.Parameter(torch.zeros(1, hidden))

    def _step(self, inputs, h, c):
        return self.lstm(inputs, (h, c))

    def _sample_from(self, logits, n_valid, h):
        mask = torch.full_like(logits, float("-inf"))
        mask[:n_valid] = logits[:n_valid]
        probs = F.softmax(mask, dim=0)
        dist = torch.distributions.Categorical(probs[:n_valid])
        idx = dist.sample()
        return idx.item(), dist.log_prob(idx), dist.entropy(), F.one_hot(idx.long(), self.input_size).float().view(1, -1)

    def sample_fpn_arch(self):
        h, c = self.h0.clone(), self.c0.clone()
        inputs = torch.zeros(1, self.input_size, device=h.device)
        fpn_arch, log_probs, entropies = [], [], []
        pool_size = 3

        for _ in range(N_FPN_BLOCKS):
            block_cfg = {}
            for key, dec, n_choices, vocab in [
                ("id1", self.dec_id, pool_size, None),
                ("id2", self.dec_id, pool_size, None),
                ("op1", self.dec_op, len(FPN_OPS), FPN_OPS),
                ("op2", self.dec_op, len(FPN_OPS), FPN_OPS),
                ("agg", self.dec_agg, len(AGG_OPS), AGG_OPS)
            ]:
                h, c = self._step(inputs, h, c)
                idx, lp, ent, inputs = self._sample_from(dec(h).squeeze(0), n_choices, h)
                block_cfg[key] = vocab[idx] if vocab else idx
                log_probs.append(lp); entropies.append(ent)
            fpn_arch.append(block_cfg)
            pool_size += 1
        return fpn_arch, log_probs, entropies

    def sample_head_arch(self):
        h, c = self.h0.clone(), self.c0.clone()
        inputs = torch.zeros(1, self.input_size, device=h.device)
        head_arch, log_probs, entropies = [], [], []

        for _ in range(N_HEAD_OPS):
            h, c = self._step(inputs, h, c)
            idx, lp, ent, inputs = self._sample_from(self.dec_hop(h).squeeze(0), len(HEAD_OPS), h)
            head_arch.append(HEAD_OPS[idx])
            log_probs.append(lp); entropies.append(ent)

        h, c = self._step(inputs, h, c)
        share_from, lp_s, ent_s, _ = self._sample_from(self.dec_share(h).squeeze(0), N_HEAD_OPS + 1, h)
        log_probs.append(lp_s); entropies.append(ent_s)

        return head_arch, share_from, log_probs, entropies

def reinforce_update(log_probs, entropies, reward, baseline, opt, model, entropy_coeff=0.05):
    advantage = reward - baseline
    loss = -sum(log_probs) * advantage - entropy_coeff * torch.stack(entropies).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return advantage
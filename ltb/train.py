from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
from .graph import tensorize_states
from .model import BranchingBipartiteGNN, masked_branching_loss


@dataclass(frozen=True)
class TrainResult:
    model: BranchingBipartiteGNN
    best_validation_accuracy: float
    final_loss: float


def candidate_accuracy(model, states):
    c,v,e,mask,target = tensorize_states(states)
    model.eval()
    with torch.no_grad():
        logits = model(c,v,e).masked_fill(~mask, float("-inf"))
        pred = logits.argmax(dim=1)
    return float((pred == target).float().mean().item())


def train_branching_gnn(
    train_states,
    validation_states,
    *,
    seed=42,
    epochs=20,
    batch_size=32,
    hidden_dim=64,
    layers=3,
    learning_rate=1e-3,
):
    if not train_states or not validation_states:
        raise ValueError("training and validation states must be nonempty")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BranchingBipartiteGNN(hidden_dim=hidden_dim, layers=layers)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    final_loss = float("nan")

    for epoch in range(1, epochs+1):
        order = rng.permutation(len(train_states))
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            batch = [train_states[int(i)] for i in idx]
            c,v,e,mask,target = tensorize_states(batch)
            logits = model(c,v,e)
            loss = masked_branching_loss(logits, mask, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        final_loss = float(np.mean(losses))
        acc = candidate_accuracy(model, validation_states)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs//5) == 0:
            print(f"epoch={epoch:03d} loss={final_loss:.4f} validation_expert_accuracy={acc:.3f}")

    model.load_state_dict(best_state)
    return TrainResult(model, best_acc, final_loss)

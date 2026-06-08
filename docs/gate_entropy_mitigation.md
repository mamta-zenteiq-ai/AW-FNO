# Gate-Entropy Mitigation Playbook

Trigger: `gate_entropy >= 0.65` at epoch >= 50 during AW-FNO training.

If the gate stays uniform (`α ≈ 0.5` everywhere), the adaptive routing
contributes nothing — the model is effectively running an unconstrained
average of FNO and WNO, which is the additive ablation.

## Mitigation A — Entropy penalty (preferred; minimal code change)

Add a binary-entropy *penalty* to the training loss so the optimizer is
rewarded for decisive (low-entropy) gates:

    L_total = L_data + λ_ent * mean( H(α) )

Recommended `λ_ent`: start at `0.01`, increase to `0.05` if not enough.

### Patch (apply only if needed)

`trainers/operator_trainer.py`, inside `_train_epoch`, after `loss = self.criterion(...)`:

```python
if hasattr(self, "lambda_ent") and self.lambda_ent > 0.0:
    ent = self._gate_entropy_from_last_forward()
    if not torch.isnan(ent):
        loss = loss + self.lambda_ent * ent
```

This requires capturing the gate output during the forward pass (hook
storing the per-batch alpha) and computing entropy from it.

## Mitigation B — Re-initialize gate with random weights

The current init is zero-weight + sigmoid → α = 0.5 exactly at init.
This is a *saddle point* of the entropy landscape — escape is slow.

Patch `awfno/models/awfno.py`, `AdaptiveGatedFusion2d.__init__`:

```python
# Instead of zero init, use small random init to break symmetry
nn.init.normal_(GateConv.weight, mean=0.0, std=0.1)
nn.init.constant_(GateConv.bias, 0)
```

## Mitigation C — Richer gate (3×3 conv, 2 layers)

Give the gate spatial context so it can learn region-aware routing:

```python
self.gate = nn.Sequential(
    nn.Conv2d(channels*2, channels, kernel_size=3, padding=1),
    nn.GELU(),
    nn.Conv2d(channels, channels, kernel_size=1),
    nn.Sigmoid(),
)
```

Adds ~6× more gate parameters but no change to FNO/WNO branches.

## Decision tree

```
epoch 50: gate_H < 0.65 ?
├─ YES → continue training, no action
└─ NO  → epoch 100: gate_H < 0.60 ?
         ├─ YES → continue (slowly decisive)
         └─ NO  → STOP, apply Mitigation A, restart with λ_ent=0.01
                  └─ epoch 100 again: gate_H < 0.50 ?
                     ├─ YES → continue
                     └─ NO  → bump λ_ent to 0.05, restart
                              └─ if still flat: combine A+B
```

Goal: by epoch 200, gate_H should be < 0.5.  By epoch 500, < 0.3.

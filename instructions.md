# Masked-DPPO Implementation Spec (Option A — Through the Policy Network)

## Goal

Add an auxiliary masked action reconstruction loss to DPPO fine-tuning, where the reconstruction runs through the policy's own diffusion noise predictor `eps_theta`. The masking loss and the PPO loss share the same network parameters, so the reconstruction objective directly shapes the policy's learned representations.

We are testing: **Does an auxiliary masked reconstruction objective improve sample efficiency during DPPO fine-tuning on MuJoCo locomotion tasks?**

Any outcome (helps, hurts, or no effect) is a valid result.

---

## Codebase

Repo: `https://github.com/irom-princeton/dppo` (clone `main` branch, tag `v0.8`)

### Files you MUST read first (before writing any code)

Read these in order. Understand the data flow before touching anything.

1. `model/diffusion/diffusion.py` — Base diffusion model class. Contains:
   - `p_mean_std(x, t, cond)`: given noisy action `x` at diffusion timestep `t` and conditioning `cond` (the state), returns the predicted mean and std of the denoised action at timestep `t-1`. This calls `eps_theta` internally.
   - `p_sample(x, t, cond)`: samples `x_{t-1}` from `p_mean_std`.
   - `p_losses(x_start, cond, t)`: the training loss — adds noise to `x_start` at level `t`, predicts the noise with `eps_theta`, returns MSE between predicted and actual noise. This is used during pre-training (behavior cloning).
   - The noise schedule (`betas`, `alphas_cumprod`, etc.) and how `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps` works.
   - The network architecture (MLP or UNet) that implements `eps_theta`. It's stored as `self.model` or `self.network` — find the exact attribute name.

2. `model/diffusion/diffusion_vpg.py` — Vanilla Policy Gradient wrapper. Adds:
   - Action sampling that records the full denoising chain for log-prob computation.
   - `get_logprobs(cond, chains)`: recomputes log-probs of stored denoising chains under current parameters.

3. `model/diffusion/diffusion_ppo.py` — PPO wrapper (inherits from vpg). Adds:
   - `loss(...)` method: computes the clipped PPO surrogate loss. **This is the method you will modify.**
   - Clipping, advantage weighting, denoising discount factor application.

4. `agent/finetune/train_ppo_diffusion_agent.py` — Training loop. Contains:
   - Rollout collection: steps the environment, calls the policy, stores transitions.
   - PPO update loop: samples minibatches, calls `model.loss(...)`, backpropagates.
   - **Find where `loss.backward()` is called and where the optimizer steps.** That's where you integrate.

5. `cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_mlp.yaml` — Example config.

### Files you will modify

- `model/diffusion/diffusion_ppo.py` — Add mask reconstruction loss computation.
- `agent/finetune/train_ppo_diffusion_agent.py` — Add mask loss to total loss, add logging.
- `cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_mlp.yaml` (and walker2d, halfcheetah) — Add masking hyperparameters.

---

## Detailed Implementation

### Step 0: Understand the shapes

Before writing anything, insert print statements or breakpoints in the PPO update loop and figure out:

```
- What is the shape of the observation/state tensor? Likely (batch, obs_dim).
- What is the shape of the action chunk a^0? Likely (batch, horizon_steps * act_dim) or (batch, horizon_steps, act_dim).
  horizon_steps = 4 for Gym tasks. act_dim = 3 for Hopper, 6 for Walker2d/HalfCheetah.
- How is the denoising chain stored? Is it a list of tensors [a^K, a^{K-1}, ..., a^0]?
  Or a single tensor of shape (batch, K+1, horizon_steps * act_dim)?
- What exactly is passed to model.loss()? What are the argument names and shapes?
```

Print these. Write them down. Do NOT proceed until you know the exact shapes.

### Step 1: Extract `a^0` (the final clean actions) in the PPO update

During the PPO update, you need access to the final denoised action chunk `a^0_t` — the clean actions that were actually executed in the environment.

The denoising chain stored in the rollout buffer should contain `a^0` as the last element (denoising step k=0). Find where this is stored. It's likely accessible as something like:

```python
# Somewhere in the rollout buffer or passed to model.loss()
# The chain might be stored as: chains[i] = a^{K-i} for i in 0..K
# So chains[-1] or chains[K] = a^0, the final clean action
a0 = chains[:, -1]  # or however the indexing works — VERIFY THIS
```

If `a^0` isn't directly accessible, you can reconstruct it: during rollout collection, `a^0` is the action that gets executed in the environment. Look for where `env.step(action)` is called — `action` is (or is derived from) `a^0`. Store it explicitly in the buffer if needed.

### Step 2: Implement mask reconstruction loss in `diffusion_ppo.py`

Add a method to the PPO diffusion model class. Here is the logic:

```python
def compute_mask_loss(self, a0, cond, mask_ratio=0.5):
    """
    Compute masked action reconstruction loss through the policy's noise predictor.
    
    Args:
        a0: (batch, horizon_steps * act_dim) — the final denoised action chunks from rollouts.
             NOTE: check if this is (batch, horizon_steps * act_dim) or (batch, horizon_steps, act_dim).
             The diffusion model likely works with the flattened version. Verify and reshape as needed.
        cond: dict or tensor — the observation/state conditioning. Same format as used in p_losses.
        mask_ratio: float — fraction of action-timesteps to mask within each chunk.
    
    Returns:
        loss_mask: scalar — MSE reconstruction loss over masked positions only.
    """
    batch_size = a0.shape[0]
    
    # --- 1. Create the temporal mask ---
    # We mask at the level of action-TIMESTEPS within the chunk, not individual dimensions.
    # For horizon_steps=4, act_dim=3: we mask entire timesteps (all 3 dims at once).
    #
    # a0 is likely shape (batch, horizon_steps * act_dim). We need to know horizon_steps and act_dim.
    # These should be available as self.horizon_steps and self.action_dim or similar — find them.
    
    horizon_steps = self.horizon_steps  # VERIFY attribute name
    act_dim = self.action_dim           # VERIFY attribute name
    
    n_mask = max(1, int(mask_ratio * horizon_steps))  # number of timesteps to mask
    n_mask = min(n_mask, horizon_steps - 1)            # always keep at least 1 visible
    
    # For each sample in the batch, randomly choose which timesteps to mask
    # Create mask of shape (batch, horizon_steps)
    mask = torch.zeros(batch_size, horizon_steps, device=a0.device)
    for i in range(batch_size):
        indices = torch.randperm(horizon_steps, device=a0.device)[:n_mask]
        mask[i, indices] = 1.0
    
    # Expand mask to cover all action dimensions: (batch, horizon_steps * act_dim)
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, act_dim).reshape(batch_size, -1)
    # mask_expanded[i, j] = 1.0 means position j is masked for sample i
    
    # --- 2. Create masked action input ---
    a0_masked = a0.clone()
    a0_masked[mask_expanded.bool()] = 0.0
    
    # --- 3. Run through the diffusion model's noise predictor ---
    # 
    # The idea: 
    #   - Pick a LOW noise level (small k, e.g., k=1). We want a nearly-clean input.
    #   - Construct a noisy version of a0_masked at this noise level.
    #   - Ask eps_theta to predict the noise, which implicitly predicts the clean actions.
    #   - Reconstruct the predicted clean actions and compare to the real a0 at masked positions.
    #
    # Why a low noise level? We want the model to see mostly-clean actions with some positions
    # zeroed out, and predict what should be there. High noise levels would drown out the masking
    # signal in Gaussian noise.
    
    # Pick diffusion timestep — use a low value. 
    # The diffusion timesteps are integers from 0 to K-1 (or 1 to K, check convention).
    # We want k near 0 (clean end). Try k=0 or k=1.
    # IMPORTANT: Check how p_losses indexes timesteps. Some implementations use t=0 as clean,
    # others use t=0 as one step of noise. Read the code.
    
    t = torch.ones(batch_size, device=a0.device, dtype=torch.long)  # t=1, low noise
    # If the model uses 0-indexed timesteps where 0 = first noise step, use t=0.
    # If K=20 denoising steps and only last 10 are fine-tuned, make sure t=1 is within
    # the fine-tuned range. Otherwise the gradients won't flow to the fine-tuned parameters.
    # CHECK: what is ft_denoising_steps? Use t = 0 or 1 within that range.
    
    # Add noise at level t to the MASKED actions (not the original)
    noise = torch.randn_like(a0_masked)
    
    # Standard DDPM forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    # These coefficients should be available as self.sqrt_alphas_cumprod[t] etc.
    # VERIFY the exact attribute names by reading diffusion.py
    sqrt_alpha_bar = extract(self.sqrt_alphas_cumprod, t, a0_masked.shape)    # VERIFY name
    sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, a0_masked.shape)  # VERIFY name
    
    a_noisy = sqrt_alpha_bar * a0_masked + sqrt_one_minus * noise
    
    # --- 4. Predict noise using the policy network ---
    # This is the KEY step — we run eps_theta on the masked+noisy input.
    # The model sees zeros where actions are masked and must predict noise that, when removed,
    # reconstructs reasonable actions at those positions.
    
    predicted_noise = self.model(a_noisy, t, cond=cond)  # VERIFY: how is the network called?
    # self.model might be self.network or self.actor — find the actual attribute.
    # The call signature might be self.model(x, t, local_cond=cond) or similar — VERIFY.
    
    # --- 5. Reconstruct predicted clean actions ---
    # From the predicted noise, recover the predicted x_0:
    # x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
    a0_pred = (a_noisy - sqrt_one_minus * predicted_noise) / sqrt_alpha_bar
    
    # --- 6. Compute MSE loss ONLY at masked positions ---
    diff = (a0_pred - a0) ** 2  # (batch, horizon_steps * act_dim)
    
    # Zero out the unmasked positions — we only care about reconstruction at masked positions
    masked_diff = diff * mask_expanded
    
    # Mean over masked positions only
    loss_mask = masked_diff.sum() / mask_expanded.sum()
    
    return loss_mask
```

### Step 3: Integrate into the `loss()` method or the training loop

You have two integration options. Pick whichever is cleaner given the code structure:

**Option 3a: Add to the model's `loss()` method in `diffusion_ppo.py`**

If `model.loss()` returns a dict or multiple values, add `mask_loss` as an additional return:

```python
def loss(self, obs, chains, advantages, old_logprobs, ...):
    # ... existing PPO loss computation ...
    ppo_loss = ...  # existing
    
    # Extract a0 from chains
    a0 = chains[:, -1]  # VERIFY indexing
    
    # Compute mask loss
    mask_loss = self.compute_mask_loss(a0, cond=obs, mask_ratio=self.mask_ratio)
    
    return ppo_loss, mask_loss  # modify return signature
```

**Option 3b: Compute it separately in the training loop**

In `train_ppo_diffusion_agent.py`, after computing the PPO loss:

```python
# Existing:
ppo_loss = model.loss(...)

# New:
mask_loss = model.compute_mask_loss(a0, cond=obs, mask_ratio=cfg.mask.ratio)

# Combined:
total_loss = ppo_loss + cfg.mask.lambda_ * mask_loss

total_loss.backward()
optimizer.step()
```

Option 3b is less invasive and easier to toggle on/off with a config flag.

### Step 4: Handle the gradient flow correctly

**CRITICAL**: The mask loss gradients flow through `eps_theta` (the noise predictor), which is the same network that PPO updates. This is intentional — it's the whole point of Option A. But you need to be aware of the implications:

1. **The mask loss will affect PPO training.** The noise predictor is being pulled in two directions: PPO wants it to generate high-reward actions, masking wants it to be good at reconstructing masked actions. These gradients may conflict.

2. **Do NOT backpropagate the mask loss through the denoising chain.** The mask loss operates on `a^0` directly — you take the stored clean actions, mask them, noise them slightly, and run one forward pass through `eps_theta`. This is NOT running the full K-step denoising process. It's a single forward pass through the noise predictor. This is computationally cheap.

3. **Make sure `a0` is detached from the rollout computation graph.** When you extract `a0` from the rollout buffer, it should already be detached (stored as a numpy array or detached tensor). If not, call `.detach()` before using it in the mask loss to prevent gradients flowing back through the rollout.

```python
a0 = a0.detach()  # ensure no gradient flow back through rollout
```

### Step 5: Config changes

Add to `cfg/gym/finetune/hopper-v2/ft_ppo_diffusion_mlp.yaml`:

```yaml
mask:
  enabled: true
  lambda_: 0.1       # weight of mask loss relative to PPO loss
  ratio: 0.5         # fraction of timesteps to mask (2 out of 4)
  noise_level: 1     # diffusion timestep t used for the reconstruction. 
                      # low = nearly clean input, high = more noise.
                      # start with 1. if gradients don't flow (because t=1 is outside 
                      # ft_denoising_steps range), try the lowest fine-tuned step.
```

Create a baseline config that's identical but with `mask.enabled: false` (or just `mask.lambda_: 0.0`).

### Step 6: Logging

Add these to WandB logging alongside existing metrics:

```python
if cfg.mask.enabled:
    log_dict["loss/mask_reconstruction"] = mask_loss.item()
    log_dict["loss/mask_weighted"] = (cfg.mask.lambda_ * mask_loss).item()
    log_dict["loss/total_with_mask"] = total_loss.item()
```

---

## Understanding What This Actually Does to the Network

When you run the mask loss, here is what happens to `eps_theta` at a mechanistic level:

1. You take clean actions `a0 = [a_0, a_1, a_2, a_3]` (4 timesteps for Hopper).
2. You zero out, say, `a_1` and `a_3`: `a0_masked = [a_0, 0, a_2, 0]`.
3. You add a tiny amount of diffusion noise (t=1) to get `a_noisy`.
4. You ask `eps_theta(a_noisy, t=1, s_t)` to predict the noise.
5. From the predicted noise, you reconstruct `a0_pred`.
6. You compute MSE between `a0_pred` and `a0` at positions 1 and 3.
7. The gradient says: "adjust eps_theta's weights so that when it sees zeros at positions 1 and 3, it predicts noise values that reconstruct the correct actions at those positions."

**What this teaches the network:** "Given the state and a partial action sequence with gaps, infer the missing actions." This is a conditional reconstruction task, similar to inpainting in image diffusion models.

**Why this might help:** If the network gets better at understanding how actions within a chunk relate to each other and to the state, it might generate more internally-consistent chunks, which could lead to more efficient credit assignment during PPO updates.

**Why this probably won't help:** The network already generates all 4 actions jointly from the state via the full denoising process. The denoising process itself is a form of iterative reconstruction from corrupted input — adding another reconstruction objective on top is redundant. Also, with only 4 timesteps, the reconstruction task is trivial: given the state and 2 actions, predicting the other 2 is a simple interpolation that doesn't require "deep structural understanding."

**Why this might actively hurt:** The mask loss gradient competes with the PPO gradient. PPO says "adjust weights to increase probability of high-reward action chunks." Mask loss says "adjust weights to better reconstruct arbitrary action chunks regardless of reward." These objectives pull in different directions. If lambda is too high, the mask loss dominates and prevents effective policy improvement.

---

## Experiment Plan

### Phase 1: Get baseline working (Day 1-2)

1. Clone repo, install dependencies for Gym tasks.
2. Download pre-trained checkpoints (the repo does this automatically).
3. Run baseline DPPO fine-tuning on Hopper-v2 with default config. Single seed.
4. Verify the reward curve roughly matches the paper's Figure 5.
5. Record: final reward, convergence speed, wall-clock time per iteration.

### Phase 2: Implement and sanity-check masking (Day 3-5)

1. Implement `compute_mask_loss` as described above.
2. Run with `mask.lambda_: 0.0` first — verify that the mask loss is computed and logged but doesn't affect training (reward curves should match baseline exactly).
3. Run with `mask.lambda_: 0.1`. Check:
   - Does it crash? (shape mismatches, gradient errors)
   - Does mask_loss decrease over training? (if not, something is wrong)
   - Does reward still increase? (if it collapses, reduce lambda)

### Phase 3: Lambda sweep on Hopper (Day 6-8)

Run 3 seeds each:
- Baseline DPPO (lambda=0)
- Masked-DPPO lambda=0.01
- Masked-DPPO lambda=0.1
- Masked-DPPO lambda=1.0

All with mask_ratio=0.5.

Pick the lambda that works best (or least-badly). If all are the same as baseline, use lambda=0.1 for the remaining experiments.

### Phase 4: Three-environment comparison (Day 9-14)

Using best lambda from Phase 3, run on all three Gym tasks:
- Hopper-v2: 3-5 seeds, baseline and masked
- Walker2D-v2: 3-5 seeds, baseline and masked  
- HalfCheetah-v2: 3-5 seeds, baseline and masked

### Phase 5: Analysis and writing (Day 15-21)

Generate:
1. **Main figure**: Reward vs. Environment Steps for each of the 3 tasks, baseline vs. masked, with mean ± std shading. This is the most important figure.
2. **Mask loss curve**: How does the reconstruction loss evolve over training? Does it decrease? Plateau?
3. **Lambda ablation table**: Final reward for each lambda value on Hopper.
4. **Analysis section**: Why did masking help / not help / hurt? Reference the specific structural differences:
   - Short chunks (4 timesteps) vs. long sequences in NLP (512 tokens)
   - Self-generated targets vs. fixed corpus in BERT
   - Denoising already does corruption+reconstruction
   - Gradient competition between PPO and reconstruction objectives
   - MGP uses masking at inference time (ATR), not just training time

---

## Potential Issues and Fixes

### "t=1 is outside the fine-tuned denoising steps range"

DPPO often only fine-tunes the last K' denoising steps (e.g., last 10 out of 20). The diffusion timestep `t` in the mask loss must be within the fine-tuned range, otherwise gradients won't flow to the parameters being updated.

Check: what is `ft_denoising_steps` in the config? If the model fine-tunes steps k=0 through k=9 (the last 10 denoising steps), then t=1 in the mask loss corresponds to a step that IS being fine-tuned. But verify the indexing convention — does k=0 correspond to the cleanest step or the noisiest fine-tuned step?

If gradients aren't flowing (mask_loss doesn't decrease, or `.grad` is None on the fine-tuned parameters), this is the likely cause. Try different values of `t` within the fine-tuned range.

### "The mask loss is trivially zero from the start"

With horizon_steps=4 and mask_ratio=0.5, you mask 2 timesteps and leave 2 visible. The network sees the state + 2 clean actions + small noise. Predicting the other 2 actions might be extremely easy, especially since the state alone might determine the full chunk.

If this happens, try:
- mask_ratio=0.75 (mask 3 out of 4 — harder)
- Remove state conditioning from the mask loss (only give masked actions, not obs). This makes reconstruction harder and forces the model to rely on inter-action dependencies rather than state→action mapping. BUT this requires modifying how you call eps_theta, which expects conditioning.

### "The PPO loss explodes after adding mask loss"

The mask loss gradient might be much larger in magnitude than the PPO gradient, destabilizing training.

Fix: reduce lambda. Or gradient-clip the mask loss gradient separately before adding to the PPO gradient. The simplest approach:

```python
mask_loss = model.compute_mask_loss(...)
scaled_mask_loss = cfg.mask.lambda_ * mask_loss
# Compute gradients separately if needed
```

### "I can't figure out how to call eps_theta with masked inputs"

The noise predictor `eps_theta` is the neural network (MLP or UNet) stored as an attribute of the diffusion model. It takes `(x, t, cond)` where `x` is the noisy action, `t` is the diffusion timestep, and `cond` is the observation. Find the exact call signature by reading `p_losses` in `diffusion.py` — it calls the network there for the standard BC training loss. Mimic that call pattern.

```python
# In p_losses you'll find something like:
noise_pred = self.model(x_noisy, t, local_cond=cond, global_cond=None)
# or
noise_pred = self.network(x_noisy, t, cond)
# Copy exactly this calling convention for your mask loss.
```

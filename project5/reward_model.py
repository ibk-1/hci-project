# project5/reward_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .policy import PolicyNet  # used for KL reference in RLHF

class TrajRewardNet(nn.Module):
    """
    Per-step scorer; trajectory score = sum of step scores.
    Inputs per step: state [6,5,5] and one-hot action [4].
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5 + 4, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward_step(self, state, action_onehot):
        # state: [B,6,5,5], action_onehot: [B,4]
        feat = self.conv(state.float())           # [B, 16*25]
        x = torch.cat([feat, action_onehot], -1)  # [B, 16*25+4]
        return self.fc(x)                         # [B,1]

    def traj_score(self, states, actions):
        """
        states:  [T,6,5,5] float32
        actions: [T] int64
        Returns scalar sum of step scores.
        """
        T = states.shape[0]
        aoh = F.one_hot(actions, num_classes=4).float()   # [T,4]
        s = self.forward_step(states, aoh).view(T)        # [T]
        return s.sum()

def bt_train_reward(reward_net: TrajRewardNet, pairs, epochs=200, lr=1e-3, device="cpu"):
    """
    Bradley–Terry training from pairwise prefs.
    pairs: list of dicts with keys:
      - traj_a: {"states": (T,6,5,5) float32, "actions": (T,) int64}
      - traj_b: {...}
      - pref: 0 if A preferred, 1 if B preferred
    """
    reward_net = reward_net.to(device)
    opt = optim.Adam(reward_net.parameters(), lr=lr)

    for ep in range(epochs):
        total = 0.0
        for p in pairs:
            sa = torch.from_numpy(p["traj_a"]["states"]).to(device)
            aa = torch.from_numpy(p["traj_a"]["actions"]).to(device)
            sb = torch.from_numpy(p["traj_b"]["states"]).to(device)
            ab = torch.from_numpy(p["traj_b"]["actions"]).to(device)

            Ra = reward_net.traj_score(sa, aa)
            Rb = reward_net.traj_score(sb, ab)

            # P(A>B) = sigmoid(Ra - Rb)
            logit = Ra - Rb
            y = torch.tensor(0.0 if p["pref"] == 0 else 1.0, device=device)
            loss = F.binary_cross_entropy_with_logits(logit, y)

            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())

        if (ep + 1) % 50 == 0:
            print(f"[BT] epoch {ep+1}/{epochs} loss={total/len(pairs):.3f}")
    return reward_net

def reinforce_with_learned_reward(policy: PolicyNet, reward_net: TrajRewardNet,
                                  kl_ref: PolicyNet, beta=0.01, epochs=30, N=16,
                                  lr=3e-3, gamma=0.99, seed=0, device="cpu"):
    """
    Fine-tune policy with learned reward and a KL penalty to the reference policy.
    Loss ~ -(sum learned reward + log-likelihood) + beta * KL(pi || pi_ref)
    """
    from .rl_train import rollout  # avoid circular import at module import time
    policy = policy.to(device); kl_ref = kl_ref.to(device); reward_net = reward_net.to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)
    torch.manual_seed(seed); np.random.seed(seed)

    for ep in range(epochs):
        total_loss = 0.0
        for n in range(N):
            traj = rollout(policy, gamma=gamma, rng=seed + ep*100 + n)
            S = torch.from_numpy(traj["states"]).to(device)   # [T,6,5,5]
            A = torch.from_numpy(traj["actions"]).to(device)  # [T]

            # learned reward for the whole trajectory
            Rhat = reward_net.traj_score(S, A)

            # log-likelihood bonus (encourages actions taken)
            logits = policy.forward(S.float())
            logp = F.log_softmax(logits, dim=-1)[torch.arange(len(A)), A].mean()

            # KL(pi || pi_ref) averaged over states
            with torch.no_grad():
                ref = F.log_softmax(kl_ref.forward(S.float()), dim=-1)  # [T,4]
            cur = F.log_softmax(logits, dim=-1)
            kl = torch.sum(torch.exp(cur) * (cur - ref), dim=-1).mean()

            loss = -(Rhat + logp) + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())

        if (ep + 1) % 5 == 0:
            print(f"[RLHF] epoch {ep+1}/{epochs} loss={total_loss/N:.3f}")
    return policy

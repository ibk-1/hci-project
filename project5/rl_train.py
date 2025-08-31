import torch, torch.nn.functional as F, torch.optim as optim
import numpy as np
from .rl_env import MouseGrid
from .policy import PolicyNet

@torch.no_grad()
def rollout(policy: PolicyNet, max_steps=40, gamma=0.99, rng=None):
    env = MouseGrid(rng=rng, max_steps=max_steps)
    obs = env.reset()
    states, actions, rewards, infos = [], [], [], []
    done = False
    while not done:
        x = torch.from_numpy(obs[None, ...])  # [1,6,5,5]
        logits = policy.forward(x.float())
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        a = np.random.choice(4, p=probs)
        sr = env.step(int(a))
        states.append(obs)
        actions.append(a)
        rewards.append(sr.reward)
        infos.append(sr.info | {"grid": env.render_ascii()})
        obs = sr.obs
        done = sr.done
    # returns-to-go
    G = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma*g
        G.append(g)
    G = list(reversed(G))
    return {"states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "returns": np.array(G, dtype=np.float32),
            "infos": infos}

def reinforce_train(epochs=50, N=16, lr=3e-3, gamma=0.99, seed=0, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    policy = PolicyNet().to(device)
    opt = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        batch_logp = []
        batch_rtgs = []
        total_return = 0.0

        for n in range(N):
            traj = rollout(policy, gamma=gamma, rng=seed+ep*100+n)
            states = torch.from_numpy(traj["states"]).to(device)           # [T,6,5,5]
            actions = torch.from_numpy(traj["actions"]).to(device)         # [T]
            returns = torch.from_numpy(traj["returns"]).to(device)         # [T]
            logits = policy.forward(states.float())                        # [T,4]
            logp = F.log_softmax(logits, dim=-1)[torch.arange(len(actions)), actions]
            batch_logp.append(logp)
            batch_rtgs.append(returns)
            total_return += float(sum(traj["rewards"]))

        logp = torch.cat(batch_logp)
        rtgs = torch.cat(batch_rtgs)
        # normalize returns helps stability
        rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-8)

        loss = -(logp * rtgs).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if (ep+1) % 5 == 0:
            print(f"[REINFORCE] epoch {ep+1}/{epochs} loss={loss.item():.3f} avg_return={total_return/(N):.2f}")

    return policy

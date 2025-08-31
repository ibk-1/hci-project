import torch, numpy as np
from django.core.management.base import BaseCommand
from project5.models import Preference
from project5.reward_model import TrajRewardNet, bt_train_reward

class Command(BaseCommand):
    help = "Train Bradley–Terry reward model from stored preferences"

    def handle(self, *args, **opts):
        prefs = Preference.objects.exclude(choice__isnull=True)[:200]
        if not prefs:
            self.stderr.write("No labeled preferences found.")
            return
        pairs=[]
        for p in prefs:
            pairs.append({
                "traj_a": {"states": np.array(p.traj_a["states"], dtype=np.float32),
                           "actions": np.array(p.traj_a["actions"], dtype=np.int64)},
                "traj_b": {"states": np.array(p.traj_b["states"], dtype=np.float32),
                           "actions": np.array(p.traj_b["actions"], dtype=np.int64)},
                "pref": int(p.choice)
            })
        net = TrajRewardNet()
        bt_train_reward(net, pairs)
        # torch.save(net.state_dict(), "project5/data/reward.pt")
        self.stdout.write(self.style.SUCCESS("Reward model trained."))

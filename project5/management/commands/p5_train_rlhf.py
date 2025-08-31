import torch
from django.core.management.base import BaseCommand
from project5.policy import PolicyNet
from project5.reward_model import TrajRewardNet, reinforce_with_learned_reward

class Command(BaseCommand):
    help = "RLHF: fine-tune policy with learned reward and KL penalty"

    def handle(self, *args, **opts):
        policy = PolicyNet()
        kl_ref = PolicyNet()
        kl_ref.load_state_dict(policy.state_dict())  # reference snapshot

        reward_net = TrajRewardNet()
        # If you saved weights: reward_net.load_state_dict(torch.load("project5/data/reward.pt"))

        reinforce_with_learned_reward(policy, reward_net, kl_ref, beta=0.01, epochs=30, N=16)
        # torch.save(policy.state_dict(), "project5/data/policy_rlhf.pt")
        self.stdout.write(self.style.SUCCESS("RLHF fine-tuning complete."))

from django.core.management.base import BaseCommand
from project5.rl_train import reinforce_train

class Command(BaseCommand):
    help = "Train baseline policy with REINFORCE"

    def add_arguments(self, parser):
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--n", type=int, default=16)

    def handle(self, *args, **opts):
        policy = reinforce_train(epochs=opts["epochs"], N=opts["n"])
        # You can persist with torch.save if you like:
        # import torch; torch.save(policy.state_dict(), "project5/data/policy_baseline.pt")
        self.stdout.write(self.style.SUCCESS("Baseline policy trained."))

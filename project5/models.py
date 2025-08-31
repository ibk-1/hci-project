
from django.db import models
from django.utils import timezone

class Preference(models.Model):
    created = models.DateTimeField(default=timezone.now)
    # store minimal trajectory footprints as JSON
    traj_a = models.JSONField()
    traj_b = models.JSONField()
    choice = models.IntegerField(null=True, blank=True)  # 0 for A, 1 for B

    def __str__(self):
        return f"Pref {self.id} choice={self.choice}"


class TrainingJob(models.Model):
    KIND_CHOICES = [
        ("reinforce", "Baseline REINFORCE"),
        ("reward", "Reward (Bradley–Terry)"),
        ("rlhf", "RLHF fine-tune"),
    ]
    kind = models.CharField(max_length=16, choices=KIND_CHOICES)
    created = models.DateTimeField(default=timezone.now)
    started = models.DateTimeField(null=True, blank=True)
    finished = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=16, default="pending")  # pending|running|done|error
    logs = models.TextField(blank=True, default="")

    def append(self, line: str):
        self.logs += (line.rstrip() + "\n")
        self.save(update_fields=["logs"])
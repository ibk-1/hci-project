from django.shortcuts import render

# Create your views here.
import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import Preference
from .policy import PolicyNet
from .rl_train import rollout
import torch
import json, threading, traceback, time
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from .models import Preference, TrainingJob
from .rl_train import reinforce_train
from .reward_model import TrajRewardNet, bt_train_reward, reinforce_with_learned_reward
from .policy import PolicyNet
import numpy as np
import torch

# keep a singleton policy in memory for quick demo
POLICY = PolicyNet()
DEVICE = "cpu"

def landing(request):
    return render(request, "project5/landing.html")

def label(request):
    return render(request, "project5/label.html")

def _rollout_to_payload(traj):
    # frames = ASCII grid per step (full episode), rewards = list of floats
    frames = [info["grid"] for info in traj["infos"]]
    rewards = [float(r) for r in traj["rewards"]]
    cells   = [int(info.get("cell", -1)) for info in traj["infos"]]
    bumped  = [bool(info.get("bumped", False)) for info in traj["infos"]]
    return {
        "total_reward": float(sum(traj["rewards"])),
        "frames": frames,
        "rewards": rewards,
        "cells": cells,
        "bumped": bumped
    }


@csrf_exempt
def api_pair(request):
    """Return two trajectories from current policy for human preference."""
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    # two different random rollouts
    t1 = rollout(POLICY, rng=None)
    t2 = rollout(POLICY, rng=None)
    pay_a = _rollout_to_payload(t1)
    pay_b = _rollout_to_payload(t2)

    pref = Preference.objects.create(
        traj_a={"states": t1["states"].tolist(), "actions": t1["actions"].tolist()},
        traj_b={"states": t2["states"].tolist(), "actions": t2["actions"].tolist()},
        choice=None
    )
    return JsonResponse({"id": pref.id, "a": pay_a, "b": pay_b})

@csrf_exempt
def api_choose(request):
    """Record preference: 0 means A preferred, 1 means B preferred."""
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    body = json.loads(request.body.decode("utf-8"))
    pid = int(body["id"]); choice = int(body["choice"])
    pref = Preference.objects.get(id=pid)
    pref.choice = 0 if choice == 0 else 1
    pref.save()
    return JsonResponse({"ok": True})


def trainer(request):
    jobs = TrainingJob.objects.order_by("-created")[:10]
    return render(request, "project5/trainer.html", {"jobs": jobs})

def _run_in_thread(job: TrainingJob, target, *args, **kwargs):
    def _wrapper():
        try:
            job.status = "running"; job.started = timezone.now(); job.save(update_fields=["status","started"])
            target(job, *args, **kwargs)
            job.status = "done"; job.finished = timezone.now(); job.save(update_fields=["status","finished"])
        except Exception:
            job.append("\n" + traceback.format_exc())
            job.status = "error"; job.finished = timezone.now(); job.save(update_fields=["status","finished"])
    t = threading.Thread(target=_wrapper, daemon=True)
    t.start()

# ---- job bodies with simple logging ----------------------------------------

def _train_reinforce(job: TrainingJob, epochs=20, n=8):
    job.append(f"[start] REINFORCE epochs={epochs} N={n}")
    policy = reinforce_train(epochs=epochs, N=n)  # prints to stdout too
    # torch.save(policy.state_dict(), "project5/data/policy_baseline.pt")
    job.append("[done] baseline policy trained")

def _train_reward(job: TrainingJob):
    prefs = Preference.objects.exclude(choice__isnull=True)[:500]
    if not prefs:
        job.append("No labeled preferences found. Go to /p5/label and collect A/B choices.")
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
    job.append(f"[info] training reward on {len(pairs)} pairs")
    bt_train_reward(net, pairs)
    # torch.save(net.state_dict(), "project5/data/reward.pt")
    job.append("[done] reward model trained")

def _train_rlhf(job: TrainingJob, beta=0.01, epochs=20, n=8):
    job.append(f"[start] RLHF beta={beta} epochs={epochs} N={n}")
    policy = PolicyNet()
    kl_ref = PolicyNet()
    kl_ref.load_state_dict(policy.state_dict())
    reward_net = TrajRewardNet()
    reinforce_with_learned_reward(policy, reward_net, kl_ref, beta=beta, epochs=epochs, N=n)
    # torch.save(policy.state_dict(), "project5/data/policy_rlhf.pt")
    job.append("[done] RLHF fine-tuning complete")

# ---- endpoints --------------------------------------------------------------

@csrf_exempt
def api_train_start(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    body = json.loads(request.body.decode("utf-8"))
    kind = body.get("kind")
    if kind not in {"reinforce","reward","rlhf"}:
        return HttpResponseBadRequest("unknown kind")

    job = TrainingJob.objects.create(kind=kind, status="pending")
    if kind == "reinforce":
        _run_in_thread(job, _train_reinforce, epochs=int(body.get("epochs",20)), n=int(body.get("n",8)))
    elif kind == "reward":
        _run_in_thread(job, _train_reward)
    else:
        _run_in_thread(job, _train_rlhf, beta=float(body.get("beta",0.01)),
                       epochs=int(body.get("epochs",20)), n=int(body.get("n",8)))
    return JsonResponse({"id": job.id})

def api_train_status(request, job_id: int):
    job = TrainingJob.objects.get(id=job_id)
    return JsonResponse({
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "logs": job.logs,
        "started": job.started.isoformat() if job.started else None,
        "finished": job.finished.isoformat() if job.finished else None,
    })


from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import socket
import getpass


import torch
import torch.nn as nn

import tasks
import models

device = "cuda" if torch.cuda.is_available() else "cpu"


PATH_REGISTRY = {
    "atlandau": Path("/n/home05/atlandau/GitHub/mcn2024-psets/jobs/"),
    "landauland": Path("/Users/landauland/Documents/Kempner/MCN2024/jobs/"),
}


def get_hostname():
    return socket.gethostname()


def get_username():
    return getpass.getuser()


hostname = get_hostname()
hostname = hostname if hostname in PATH_REGISTRY else get_username()
if hostname not in PATH_REGISTRY:
    raise ValueError(f"hostname ({hostname}) is not registered in the path registry")
filepath = PATH_REGISTRY[hostname]


def load_job(job_id, model_index=None):
    suffix = "" if model_index is None else f"_{model_index}"
    model = torch.load(filepath / f"{job_id}" / f"model{suffix}.pt", map_location=device)
    results = torch.load(filepath / f"{job_id}" / f"results{suffix}.pt", map_location=device)
    return model, results


def measure_choice(task, output):
    start_decision = task.stim_time + task.delay_time
    evidence = output[:, start_decision:].sum(dim=1)
    choice = 1 * (evidence[:, 0] < evidence[:, 1])
    return choice, evidence


@torch.no_grad()
def test_and_perturb(net, task, perturb_ratio=0.1, perturb_target="intrinsic", num_trials=100, verbose=False):
    pnet = deepcopy(net)
    pnet.eval()

    if perturb_target == "intrinsic":
        base_hidden_gain = pnet.hidden_gain.data.clone()
        base_hidden_threshold = pnet.hidden_threshold.data.clone()
    elif perturb_target == "receptive":
        base_reccurent_receptive = pnet.reccurent_receptive.data.clone()
    elif perturb_target == "projective":
        base_reccurent_projective = pnet.reccurent_projective.data.clone()
    else:
        raise ValueError(f"Unknown perturb target: {perturb_target}, recognized: 'intrinsic', 'receptive', 'projective'")

    loss = torch.zeros(num_trials)
    progress = tqdm(range(num_trials)) if verbose else range(num_trials)
    for trial in progress:
        if perturb_target == "intrinsic":
            pnet.hidden_gain.data = base_hidden_gain + torch.randn_like(base_hidden_gain) * perturb_ratio
            pnet.hidden_threshold.data = base_hidden_threshold + torch.randn_like(base_hidden_threshold) * perturb_ratio
        elif perturb_target == "receptive":

            pnet.reccurent_receptive.data = base_reccurent_receptive + torch.randn_like(base_reccurent_receptive) * perturb_ratio
        elif perturb_target == "projective":
            pnet.reccurent_projective.data = base_reccurent_projective + torch.randn_like(base_reccurent_projective) * perturb_ratio

        X, target, _ = task.generate_data(100, source_floor=0.1)
        outputs = pnet(X, return_hidden=False)

        loss[trial] = nn.MSELoss(reduction="sum")(outputs, target).item()

    return loss


def evaluate_model(jobid, model_index, perturb_ratios, num_trials):
    # Load trained model and results
    model, results = load_job(jobid, model_index=model_index)

    args = results["args"]
    task_params = results["task"]
    train_loss = results["train_loss"]

    task = tasks.ContextualGoNogo(
        args["input_dimensions"],
        args["end_sigma"],
        num_contexts=task_params["num_contexts"],
        stim_time=task_params["stim_time"],
        delay_time=args["end_delay"],
        decision_time=task_params["decision_time"],
    )
    task.cursors = task_params["cursors"]

    net = models.GainRNN(
        task.input_dimensionality(),
        args["num_neurons"],
        task.output_dimensionality(),
        input_rank=args["input_rank"],
        recurrent_rank=args["recurrent_rank"],
    )

    net.load_state_dict(model)
    net = net.to(device)

    num_ratios = len(perturb_ratios)

    loss_intrinsic = torch.zeros(num_ratios, num_trials)
    loss_receptive = torch.zeros(num_ratios, num_trials)
    loss_projective = torch.zeros(num_ratios, num_trials)

    for i, perturb_ratio in enumerate(tqdm(perturb_ratios)):
        loss_intrinsic[i] = test_and_perturb(net, task, perturb_ratio=perturb_ratio, perturb_target="intrinsic", num_trials=num_trials)
        loss_receptive[i] = test_and_perturb(net, task, perturb_ratio=perturb_ratio, perturb_target="receptive", num_trials=num_trials)
        loss_projective[i] = test_and_perturb(net, task, perturb_ratio=perturb_ratio, perturb_target="projective", num_trials=num_trials)

    results = dict(
        jobid=jobid,
        model_index=model_index,
        args=args,
        task=task_params,
        train_loss=train_loss,
        perturb_ratios=perturb_ratios,
        loss_intrinsic=loss_intrinsic,
        loss_receptive=loss_receptive,
        loss_projective=loss_projective,
    )

    return results


if __name__ == "__main__":
    jobid = 7
    directory = filepath / f"{jobid}"

    # Get all models (from the directory, of the form f"model_{i}.pt")
    model_indices = sorted([int(f.stem.split("_")[-1]) for f in directory.glob("model_*.pt")])
    model_indices = model_indices[:2]

    # Set up perturbation analyses
    perturb_ratios = torch.logspace(-3, -1, 2)
    num_ratios = len(perturb_ratios)
    num_trials = 5

    num_models = len(model_indices)

    loss_intrinsic = torch.zeros(num_models, num_ratios, num_trials)
    loss_receptive = torch.zeros(num_models, num_ratios, num_trials)
    loss_projective = torch.zeros(num_models, num_ratios, num_trials)

    for i, model_index in enumerate(tqdm(model_indices)):
        results = evaluate_model(jobid, model_index, perturb_ratios, num_trials)
        loss_intrinsic[i] = results["loss_intrinsic"]
        loss_receptive[i] = results["loss_receptive"]
        loss_projective[i] = results["loss_projective"]

    results = dict(
        jobid=jobid,
        model_indices=model_indices,
        perturb_ratios=perturb_ratios,
        loss_intrinsic=loss_intrinsic,
        loss_receptive=loss_receptive,
        loss_projective=loss_projective,
    )

    torch.save(results, directory / "perturb_results.pt")

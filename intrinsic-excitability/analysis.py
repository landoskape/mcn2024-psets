from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import socket
import getpass

import numpy as np
from scipy.optimize import curve_fit
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


def load_job(job_id, model_index=None, init=False):
    suffix = "" if model_index is None else f"_{model_index}"
    model = torch.load(filepath / f"{job_id}" / f"model{suffix}.pt", map_location=torch.device("cpu"))
    results = torch.load(filepath / f"{job_id}" / f"results{suffix}.pt", map_location=torch.device("cpu"))
    if init:
        init_model = torch.load(filepath / f"{job_id}" / f"init_model{suffix}.pt", map_location=torch.device("cpu"))
        return model, results, init_model
    return model, results


def load_perturbation(job_id, suffix=""):
    if suffix != "" and suffix[0] != "_":
        suffix = f"_{suffix}"
    perturb = torch.load(filepath / f"{job_id}" / f"perturb_results{suffix}.pt", map_location=device)
    return perturb


def create_cdf_proportional_bins(num_bins, std_dev=1, range_multiplier=2):
    # Define the range of the distribution (e.g., ±4 standard deviations)
    lower_bound = -range_multiplier * std_dev
    upper_bound = range_multiplier * std_dev

    # Create evenly spaced probabilities
    probabilities = torch.linspace(0, 1, num_bins + 1)

    # Use the inverse CDF (icdf) to get the bin edges
    normal = torch.distributions.Normal(loc=0, scale=std_dev)
    bin_edges = normal.icdf(probabilities)

    # Clip the bin edges to the defined range
    bin_edges = torch.clamp(bin_edges, lower_bound, upper_bound)

    return bin_edges


def measure_choice(task, output):
    start_decision = task.stim_time + task.delay_time
    evidence = output[:, start_decision:].sum(dim=1)
    choice = 1 * (evidence[:, 0] < evidence[:, 1])
    return choice, evidence


def sigmoid(x, L, x0, k, b):
    """
    Sigmoid function
    L: the curve's maximum value
    x0: the x-value of the sigmoid's midpoint
    k: the steepness of the curve
    b: the y-axis intercept
    """
    return L / (1 + np.exp(-k * (x - x0))) + b


def fit_sigmoid(s, p):
    """
    Fit sigmoid function to data using curve_fit
    s: stimulus values
    p: proportion values
    """
    # Initial parameter guesses
    p0 = [max(p), np.median(s), 1, min(p)]

    # Fit the function
    popt, _ = curve_fit(sigmoid, s, p, p0, method="lm")

    return popt


@torch.no_grad()
def test_and_perturb(net, task, psychometric_edges, perturb_ratio=0.1, perturb_target="intrinsic", num_trials=100, verbose=False):
    def _update_parameter(parameter, perturb_ratio):
        update = perturb_ratio * parameter * torch.randn_like(parameter)
        return parameter + update

    pnet = deepcopy(net)
    pnet.eval()

    learning_tau = type(pnet) == models.TauRNN
    if perturb_target == "intrinsic":
        base_hidden_gain = pnet.hidden_gain.data.clone()
        if learning_tau:
            base_hidden_tau = pnet.hidden_tau.data.clone()
        else:
            base_hidden_threshold = pnet.hidden_threshold.data.clone()
    elif perturb_target == "receptive":
        base_reccurent_receptive = pnet.reccurent_receptive.data.clone()
    elif perturb_target == "projective":
        base_reccurent_projective = pnet.reccurent_projective.data.clone()
    else:
        raise ValueError(f"Unknown perturb target: {perturb_target}, recognized: 'intrinsic', 'receptive', 'projective'")

    loss = torch.zeros(num_trials)
    accuracy = torch.zeros(num_trials)
    evidence = torch.zeros(num_trials)
    fixation = torch.zeros(num_trials)
    psychometric = torch.zeros(num_trials, len(psychometric_edges) - 1)
    progress = tqdm(range(num_trials)) if verbose else range(num_trials)
    for trial in progress:
        if perturb_target == "intrinsic":
            new_gain = _update_parameter(torch.exp(base_hidden_gain), perturb_ratio)
            pnet.hidden_gain.data = torch.log(new_gain)
            if learning_tau:
                # the tau must be positive so it's passed through an exponential.
                # we want the perturb_ratio to be accurate post-exponential -- so need to wrangle a bit
                new_tau = _update_parameter(torch.exp(base_hidden_tau), perturb_ratio)
                pnet.hidden_tau.data = torch.log(new_tau)
            else:
                pnet.hidden_threshold.data = _update_parameter(base_hidden_threshold, perturb_ratio)
        elif perturb_target == "receptive":
            pnet.reccurent_receptive.data = _update_parameter(base_reccurent_receptive, perturb_ratio)
        elif perturb_target == "projective":
            pnet.reccurent_projective.data = _update_parameter(base_reccurent_projective, perturb_ratio)

        X, target, params = task.generate_data(512, source_floor=0.0)
        outputs = pnet(X.to(device), return_hidden=False)

        choice, evidence, fixation = task.analyze_response(outputs)
        choice_evidence = evidence[:, 1] - evidence[:, 0]
        choice_evidence[params["labels"] == 0] *= -1

        s_target = torch.gather(params["s_empirical"], 1, params["context_idx"].unsqueeze(1)).squeeze(1)
        s_index = torch.bucketize(s_target, psychometric_edges)
        for i in range(len(psychometric_edges) - 1):
            if torch.sum(s_index == i) > 0:
                psychometric[trial, i] = torch.mean(choice[s_index == i].float())

        loss[trial] = nn.MSELoss(reduction="sum")(outputs, target.to(device)).item()
        accuracy[trial] = torch.sum(choice == params["labels"]) / choice.size(0)
        evidence[trial] = torch.mean(choice_evidence)
        fixation[trial] = torch.mean(fixation)

    return loss, accuracy, evidence, fixation, psychometric


def evaluate_model(jobid, model_index, perturb_ratios, num_trials, psychometric_edges):
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

    if args["network_type"] == "Gain":
        model_constructor = models.GainRNN
    elif args["network_type"] == "Tau":
        model_constructor = models.TauRNN
    else:
        raise ValueError(f"Did not recognize network type -- {args['network_type']}")

    net = model_constructor(
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

    accuracy_intrinsic = torch.zeros(num_ratios, num_trials)
    accuracy_receptive = torch.zeros(num_ratios, num_trials)
    accuracy_projective = torch.zeros(num_ratios, num_trials)

    evidence_intrinsic = torch.zeros(num_ratios, num_trials)
    evidence_receptive = torch.zeros(num_ratios, num_trials)
    evidence_projective = torch.zeros(num_ratios, num_trials)

    fixation_intrinsic = torch.zeros(num_ratios, num_trials)
    fixation_receptive = torch.zeros(num_ratios, num_trials)
    fixation_projective = torch.zeros(num_ratios, num_trials)
    psychometric_intrinsic = torch.zeros(num_ratios, num_trials, len(psychometric_edges) - 1)
    psychometric_receptive = torch.zeros(num_ratios, num_trials, len(psychometric_edges) - 1)
    psychometric_projective = torch.zeros(num_ratios, num_trials, len(psychometric_edges) - 1)

    for i, perturb_ratio in enumerate(tqdm(perturb_ratios)):
        c_loss, c_acc, c_ev, c_fix, c_psy = test_and_perturb(
            net, task, psychometric_edges, perturb_ratio=perturb_ratio, perturb_target="intrinsic", num_trials=num_trials
        )
        loss_intrinsic[i] = c_loss
        accuracy_intrinsic[i] = c_acc
        evidence_intrinsic[i] = c_ev
        fixation_intrinsic[i] = c_fix
        psychometric_intrinsic[i] = c_psy

        c_loss, c_acc, c_ev, c_fix, c_psy = test_and_perturb(
            net, task, psychometric_edges, perturb_ratio=perturb_ratio, perturb_target="receptive", num_trials=num_trials
        )
        loss_receptive[i] = c_loss
        accuracy_receptive[i] = c_acc
        evidence_receptive[i] = c_ev
        fixation_receptive[i] = c_fix
        psychometric_receptive[i] = c_psy

        c_loss, c_acc, c_ev, c_fix, c_psy = test_and_perturb(
            net, task, psychometric_edges, perturb_ratio=perturb_ratio, perturb_target="projective", num_trials=num_trials
        )
        loss_projective[i] = c_loss
        accuracy_projective[i] = c_acc
        evidence_projective[i] = c_ev
        fixation_projective[i] = c_fix
        psychometric_projective[i] = c_psy

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
        accuracy_intrinsic=accuracy_intrinsic,
        accuracy_receptive=accuracy_receptive,
        accuracy_projective=accuracy_projective,
        evidence_intrinsic=evidence_intrinsic,
        evidence_receptive=evidence_receptive,
        evidence_projective=evidence_projective,
        fixation_intrinsic=fixation_intrinsic,
        fixation_receptive=fixation_receptive,
        fixation_projective=fixation_projective,
        psychometric_edges=psychometric_edges,
        psychometric_intrinsic=psychometric_intrinsic,
        psychometric_receptive=psychometric_receptive,
        psychometric_projective=psychometric_projective,
    )

    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Perturbation analysis")
    parser.add_argument("--jobid", type=int, help="Job ID", default=7)
    parser.add_argument("--suffix", type=str, help="Suffix for model and results files", default="")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=100)
    parser.add_argument("--num_ratios", type=int, help="Number of perturbation ratios", default=11)
    args = parser.parse_args()

    jobid = args.jobid
    suffix = args.suffix
    if suffix != "" and suffix[0] != "_":
        suffix = f"_{suffix}"
    directory = filepath / f"{jobid}"

    # Get all models (from the directory, of the form f"model_{i}.pt")
    model_indices = sorted([int(f.stem.split("_")[-1]) for f in directory.glob("model_*.pt")])

    # Set up perturbation analyses
    perturb_ratios = torch.linspace(0, 1, args.num_ratios)
    num_ratios = len(perturb_ratios)
    num_trials = args.num_trials

    num_models = len(model_indices)

    loss_intrinsic = torch.zeros(num_models, num_ratios, num_trials)
    loss_receptive = torch.zeros(num_models, num_ratios, num_trials)
    loss_projective = torch.zeros(num_models, num_ratios, num_trials)

    accuracy_intrinsic = torch.zeros(num_models, num_ratios, num_trials)
    accuracy_receptive = torch.zeros(num_models, num_ratios, num_trials)
    accuracy_projective = torch.zeros(num_models, num_ratios, num_trials)

    evidence_intrinsic = torch.zeros(num_models, num_ratios, num_trials)
    evidence_receptive = torch.zeros(num_models, num_ratios, num_trials)
    evidence_projective = torch.zeros(num_models, num_ratios, num_trials)

    fixation_intrinsic = torch.zeros(num_models, num_ratios, num_trials)
    fixation_receptive = torch.zeros(num_models, num_ratios, num_trials)
    fixation_projective = torch.zeros(num_models, num_ratios, num_trials)

    psychometric_edges = create_cdf_proportional_bins(20, std_dev=1, range_multiplier=2)
    psychometric_intrinsic = torch.zeros(num_models, num_ratios, len(psychometric_edges) - 1)
    psychometric_receptive = torch.zeros(num_models, num_ratios, len(psychometric_edges) - 1)
    psychometric_projective = torch.zeros(num_models, num_ratios, len(psychometric_edges) - 1)

    for i, model_index in enumerate(tqdm(model_indices)):
        results = evaluate_model(jobid, model_index, perturb_ratios, num_trials, psychometric_edges)
        loss_intrinsic[i] = results["loss_intrinsic"]
        loss_receptive[i] = results["loss_receptive"]
        loss_projective[i] = results["loss_projective"]
        accuracy_intrinsic[i] = results["accuracy_intrinsic"]
        accuracy_receptive[i] = results["accuracy_receptive"]
        accuracy_projective[i] = results["accuracy_projective"]
        evidence_intrinsic[i] = results["evidence_intrinsic"]
        evidence_receptive[i] = results["evidence_receptive"]
        evidence_projective[i] = results["evidence_projective"]
        fixation_intrinsic[i] = results["fixation_intrinsic"]
        fixation_receptive[i] = results["fixation_receptive"]
        fixation_projective[i] = results["fixation_projective"]
        psychometric_intrinsic[i] = torch.mean(results["psychometric_intrinsic"], dim=1)
        psychometric_receptive[i] = torch.mean(results["psychometric_receptive"], dim=1)
        psychometric_projective[i] = torch.mean(results["psychometric_projective"], dim=1)

        # Save results every time it's finished so if the timeout happens we still have results...
        results = dict(
            jobid=jobid,
            model_indices=model_indices[:i],
            perturb_ratios=perturb_ratios,
            psychometric_edges=psychometric_edges,
            loss_intrinsic=loss_intrinsic[:i],
            loss_receptive=loss_receptive[:i],
            loss_projective=loss_projective[:i],
            accuracy_intrinsic=accuracy_intrinsic[:i],
            accuracy_receptive=accuracy_receptive[:i],
            accuracy_projective=accuracy_projective[:i],
            evidence_intrinsic=evidence_intrinsic[:i],
            evidence_receptive=evidence_receptive[:i],
            evidence_projective=evidence_projective[:i],
            fixation_intrinsic=fixation_intrinsic[:i],
            fixation_receptive=fixation_receptive[:i],
            fixation_projective=fixation_projective[:i],
            psychometric_intrinsic=psychometric_intrinsic[:i],
            psychometric_receptive=psychometric_receptive[:i],
            psychometric_projective=psychometric_projective[:i],
        )

        torch.save(results, directory / f"perturb_results{suffix}.pt")

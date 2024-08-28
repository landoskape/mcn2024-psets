from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import socket
import getpass

import numpy as np
import torch
import torch.nn as nn

import tasks
import models

from matplotlib import pyplot as plt


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


def load_results(job_id):
    # find all results files
    results_files = list(filepath.glob(f"{job_id}/results_*.pt"))
    results = []
    for results_file in results_files:
        results.append(torch.load(results_file, map_location=torch.device("cpu")))
    # Group train_* as a tensor from each results file
    train_loss = torch.stack([r["train_loss"] for r in results])
    train_accuracy = torch.stack([r["train_accuracy"] for r in results])
    train_evidence = torch.stack([r["train_evidence"] for r in results])
    train_fixation = torch.stack([r["train_fixation"] for r in results])
    return train_loss, train_accuracy, train_evidence, train_fixation


def equal_axes(ax):
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def load_weight_changes(job_id, hidden_only=True):
    directory = filepath / f"{job_id}"

    # Get all models (from the directory, of the form f"model_{i}.pt")
    model_indices = sorted([int(f.stem.split("_")[-1]) for f in directory.glob("model_*.pt")])

    if hidden_only:
        init_params = dict(hidden_gain=[], hidden_tau=[], hidden_threshold=[])
        final_params = dict(hidden_gain=[], hidden_tau=[], hidden_threshold=[])
    else:
        # assume they're all the same
        model, results, init = load_job(job_id, model_index=model_indices[0], init=True)
        init_params = {}
        final_params = {}
        for key in model:
            init_params[key] = []
            final_params[key] = []

    hidden_params = ["hidden_gain", "hidden_tau", "hidden_threshold"]
    for i, model_index in enumerate(model_indices):
        model, results, init = load_job(job_id, model_index=model_index, init=True)
        init_params["hidden_gain"].append(torch.sigmoid(init["hidden_gain"]))
        init_params["hidden_tau"].append(torch.sigmoid(init["hidden_tau"]) * results["args"]["tauscale"])
        init_params["hidden_threshold"].append(init["hidden_threshold"])
        final_params["hidden_gain"].append(torch.sigmoid(model["hidden_gain"]))
        final_params["hidden_tau"].append(torch.sigmoid(model["hidden_tau"]) * results["args"]["tauscale"])
        final_params["hidden_threshold"].append(model["hidden_threshold"])
        if not hidden_only:
            for key, value in init.items():
                if key not in hidden_params:
                    init_params[key].append(value)
                    final_params[key].append(model[key])

    for key in init_params:
        init_params[key] = torch.stack(init_params[key])
        final_params[key] = torch.stack(final_params[key])
    return init_params, final_params


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


def beeswarm(y, nbins=None):
    """thanks to: https://python-graph-gallery.com/509-introduction-to-swarm-plot-in-matplotlib/"""
    # Convert y to a NumPy array
    y = np.asarray(y)

    # Calculate the number of bins if not provided
    if nbins is None:
        nbins = len(y) // 6

    # Get upper and lower bounds of the data
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)

    # Calculate the size of each bin based on the number of bins
    dy = (yhi - ylo) / nbins

    # Calculate the upper bounds of each bin using linspace
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide the indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins  # List to store indices for each bin
    ybs = [0] * nbins  # List to store values for each bin
    nmax = 0  # Variable to store the maximum number of data points in a bin
    for j, ybin in enumerate(ybins):

        # Create a boolean mask for elements that are less than or equal to the bin upper bound
        f = y <= ybin

        # Store the indices and values that belong to this bin
        ibs[j], ybs[j] = i[f], y[f]

        # Update nmax with the maximum number of elements in a bin so far
        nmax = max(nmax, len(ibs[j]))

        # Update i and y by excluding the elements already added to the current bin
        f = ~f
        i, y = i[f], y[f]

    # Add the remaining elements to the last bin
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices to the data points in each bin
    dx = 1 / (nmax // 2)

    for i, y in zip(ibs, ybs):
        if len(i) > 1:

            # Determine the index to start from based on whether the bin has an even or odd number of elements
            j = len(i) % 2

            # Sort the indices in the bin based on the corresponding values
            i = i[np.argsort(y)]

            # Separate the indices into two groups, 'a' and 'b'
            a = i[j::2]
            b = i[j + 1 :: 2]

            # Assign x values to the 'a' group using positive values and to the 'b' group using negative values
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def create_stacked_trial_plot(X, targets, ibatch):
    fig, ax = plt.subplots(4, 1, figsize=(6, 4), layout="constrained")
    plt.subplots_adjust(hspace=0.1)

    labels = ["Inputs", "Fixation", "Context", "Target"]
    ymins = [-1.2, -0.6, -0.6, -0.6]

    colors = ["g", "m", "k", "g", "m", "r", "b"]
    linestyles = ["-", "-", "-", "-", "-", "-", "-"]
    axloc = [0, 0, 1, 2, 2, 3, 3]

    data = np.concatenate((X[ibatch], targets[ibatch]), axis=1)

    time = np.arange(X[ibatch].shape[0])

    for i, axis in enumerate(ax):
        # Plot the d=0 line
        axis.plot([-2, data.shape[0] + 2], [0, 0], "k--", linewidth=1)

        # Remove top and right spines
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.spines["left"].set_visible(False)

        # Set limits
        axis.set_ylim(ymins[i], 1.2)
        axis.yaxis.set_visible(False)

        # Label
        axis.text(0, 0.5, labels[i], transform=axis.transAxes, ha="right", va="center", fontsize=10)

        # Show x-axis only for the bottom subplot
        if i < len(ax) - 1:
            axis.xaxis.set_visible(False)
        else:
            axis.spines["bottom"].set_position(("outward", -10))

    # Add x-label to the bottom subplot
    ax[-1].set_xlabel("Time", fontsize=12)

    for i, iax in enumerate(axloc):
        ax[iax].plot(time, data[:, i], color=colors[i], linestyle=linestyles[i], linewidth=2)

    return fig, ax


def create_stacked_output_plot(hidden, output, ibatch, hidden_perturbed=None, output_perturbed=None):
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), layout="constrained")
    plt.subplots_adjust(hspace=0.1)

    labels = ["Hidden", "Output"]
    colors = ["r", "b"]

    time = np.arange(hidden[ibatch].shape[0])
    hidden = hidden[ibatch].numpy()
    output = output[ibatch].numpy()

    if hidden_perturbed is not None and output_perturbed is not None:
        hidden_perturbed = hidden_perturbed[ibatch]
        output_perturbed = output_perturbed[ibatch]
        plot_perturbed = True
    else:
        plot_perturbed = False

    for i, axis in enumerate(ax):
        # Plot the d=0 line
        axis.plot([-2, len(time) + 2], [0, 0], "k--", linewidth=1)

        # Remove top and right spines
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(False)
        axis.spines["left"].set_visible(False)

        # Set limits
        axis.yaxis.set_visible(False)

        # Label
        axis.text(0, 0.5, labels[i], transform=axis.transAxes, ha="right", va="center", fontsize=10)

        # Show x-axis only for the bottom subplot
        if i < len(ax) - 1:
            axis.xaxis.set_visible(False)
        else:
            axis.spines["bottom"].set_position(("outward", -10))

    # Add x-label to the bottom subplot
    ax[-1].set_xlabel("Time", fontsize=12)

    hylimdata = np.concatenate((hidden, hidden_perturbed)) if plot_perturbed else hidden
    oylimdata = np.concatenate((output, output_perturbed)) if plot_perturbed else output
    ylims_hidden = 1.1 * np.array([-1, 1]) * np.max(np.abs(hylimdata))
    ylims_output = 1.1 * np.array([-1, 1]) * np.max(np.abs(oylimdata))

    ax[0].set_ylim(ylims_hidden)
    ax[1].set_ylim(ylims_output)

    ax[0].plot(time, hidden, color="k", linewidth=1, alpha=0.7)
    ax[1].plot(time, output[:, 0], color=colors[0], linewidth=2)
    ax[1].plot(time, output[:, 1], color=colors[1], linewidth=2)

    if plot_perturbed:
        ax[0].plot(time, hidden_perturbed, color="r", linewidth=1, alpha=0.3)
        ax[1].plot(time, output_perturbed[:, 0], color=colors[0], linewidth=2, linestyle="--")
        ax[1].plot(time, output_perturbed[:, 1], color=colors[1], linewidth=2, linestyle="--")

    return fig, ax


def measure_choice(task, output, delay_time=None):
    delay_time = delay_time or task.delay_time
    start_decision = task.stim_time + delay_time
    evidence = output[:, start_decision:].sum(dim=1)
    choice = 1 * (evidence[:, 0] < evidence[:, 1])
    return choice, evidence


def measure_angle(v1, v2):
    return torch.acos(torch.sum(v1 * v2, dim=0) / (v1.norm(dim=0) * v2.norm(dim=0)))


def rotate_by_angle(v, rad, epochs=1000, lr=0.1, atol=1e-6, rtol=1e-6):
    assert (rad >= 0) and (rad <= torch.pi / 2), "Rotation angle must be between 0 and pi/2"
    if rad == 0.0:
        return v.clone()
    with torch.set_grad_enabled(True):
        if type(rad) != torch.Tensor:
            rad = torch.tensor(rad)
        vprime = torch.randn_like(v)
        vprime.requires_grad = True
        for _ in range(epochs):
            angles = measure_angle(vprime, v)
            loss = torch.sum((angles - rad) ** 2)
            loss.backward()
            with torch.no_grad():
                vprime -= lr * vprime.grad
                vprime /= vprime.norm(dim=0)
            vprime.grad.zero_()
            if torch.allclose(angles, rad, atol=atol, rtol=rtol):
                break
        vprime = vprime / vprime.norm(dim=0) * v.norm(dim=0)
    return vprime.detach()


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def perturb_parameter(parameter, perturb_ratio, perturb_type):
    if perturb_type == "random":
        update = perturb_ratio * parameter * torch.randn_like(parameter)
        return parameter + update
    elif perturb_type == "rotation":
        update = rotate_by_angle(parameter, perturb_ratio)
        return update
    else:
        raise ValueError(f"Unknown perturb type: {perturb_type}, recognized: 'random', 'rotation'")


@torch.no_grad()
def test_and_perturb(
    net,
    task,
    psychometric_edges,
    perturb_ratio=0.1,
    perturb_type="random",
    perturb_target="gain",
    num_trials=100,
    verbose=False,
):
    pnet = deepcopy(net)
    pnet.eval()

    fullrnn = type(pnet) == models.FullRNN
    if perturb_target == "gain":
        base_hidden_gain = pnet.hidden_gain.data.clone()
    elif perturb_target == "tau":
        base_hidden_tau = pnet.hidden_tau.data.clone()
    elif perturb_target == "threshold":
        base_hidden_threshold = pnet.hidden_threshold.data.clone()
    elif perturb_target == "receptive" and not fullrnn:
        base_recurrent_receptive = pnet.reccurent_receptive.data.clone()
        if base_recurrent_receptive.ndim > 1:
            assert (base_recurrent_receptive.ndim == 2) and (base_recurrent_receptive.size(1) == 1), "Wrong shape..."
    elif perturb_target == "projective" and not fullrnn:
        base_recurrent_projective = pnet.reccurent_projective.data.clone()
        if base_recurrent_projective.ndim > 1:
            assert (base_recurrent_projective.ndim == 2) and (base_recurrent_projective.size(1) == 1), "Wrong shape..."
    elif fullrnn:
        recurrent_weights = pnet.recurrent_weights.data.clone()
        base_recurrent_projective, base_recurrent_scale, base_recurrent_receptive = torch.linalg.svd(recurrent_weights)
        base_recurrent_scale = torch.diag(base_recurrent_scale)
        base_recurrent_background = base_recurrent_projective[:, 1:] @ base_recurrent_scale[1:][:, 1:] @ base_recurrent_receptive[1:]
        base_recurrent_projective = base_recurrent_projective[:, 0].view(-1, 1)
        base_recurrent_scale = base_recurrent_scale[0, 0]
        base_recurrent_receptive = base_recurrent_receptive[0].view(-1, 1)
    else:
        raise ValueError(f"Unknown perturb target: {perturb_target}, recognized: 'intrinsic', 'receptive', 'projective'")

    # Generate all the updates across trials at once to avoid the overhead of autograd
    if perturb_target == "gain":
        perturbed_gain = perturb_parameter(torch.sigmoid(base_hidden_gain).unsqueeze(1).expand(-1, num_trials), perturb_ratio, perturb_type)
        perturbed_gain = torch.clamp(perturbed_gain, 1e-6, 1 - 1e-6)
    elif perturb_target == "tau":
        perturbed_tau = perturb_parameter(torch.sigmoid(base_hidden_tau).unsqueeze(1).expand(-1, num_trials), perturb_ratio, perturb_type)
        perturbed_tau = torch.clamp(perturbed_tau, 1e-6, 1 - 1e-6)
    elif perturb_target == "threshold":
        perturbed_threshold = perturb_parameter(base_hidden_threshold.unsqueeze(1).expand(-1, num_trials), perturb_ratio, perturb_type)
    elif perturb_target == "receptive":
        perturbed_receptive = perturb_parameter(base_recurrent_receptive.expand(-1, num_trials), perturb_ratio, perturb_type)
    elif perturb_target == "projective":
        perturbed_projective = perturb_parameter(base_recurrent_projective.expand(-1, num_trials), perturb_ratio, perturb_type)
    else:
        raise ValueError("Unknown perturb target")

    loss = torch.zeros(num_trials)
    accuracy = torch.zeros(num_trials)
    evidence = torch.zeros(num_trials)
    fixation = torch.zeros(num_trials)
    psychometric = torch.zeros(num_trials, len(psychometric_edges) - 1)
    progress = tqdm(range(num_trials)) if verbose else range(num_trials)
    for trial in progress:
        if perturb_target == "gain":
            pnet.hidden_gain.data = inverse_sigmoid(perturbed_gain[:, trial])
        elif perturb_target == "tau":
            pnet.hidden_tau.data = inverse_sigmoid(perturbed_tau[:, trial])
        elif perturb_target == "threshold":
            pnet.hidden_threshold.data = perturbed_threshold[:, trial]
        elif perturb_target == "receptive":
            if fullrnn:
                new_first_rank = base_recurrent_scale * base_recurrent_projective.view(-1, 1) @ perturbed_receptive[:, trial].view(1, -1)
                pnet.recurrent_weights.data = base_recurrent_background + new_first_rank
            else:
                pnet.reccurent_receptive.data = perturbed_receptive[:, trial].view(-1, 1)
        elif perturb_target == "projective":
            if fullrnn:
                new_first_rank = base_recurrent_scale * perturbed_projective[:, trial].view(-1, 1) @ base_recurrent_receptive.view(1, -1)
                pnet.recurrent_weights.data = base_recurrent_background + new_first_rank
            else:
                pnet.reccurent_projective.data = perturbed_projective[:, trial].view(-1, 1)

        X, target, params = task.generate_data(512, source_floor=0.0)
        outputs = pnet(X.to(device), return_hidden=False)

        choice, c_evidence, c_fixation = task.analyze_response(outputs)
        choice_evidence = c_evidence[:, 1] - c_evidence[:, 0]
        choice_evidence[params["labels"] == 0] *= -1

        s_target = torch.gather(params["s_empirical"], 1, params["context_idx"].unsqueeze(1)).squeeze(1)
        s_index = torch.bucketize(s_target, psychometric_edges)
        for i in range(len(psychometric_edges) - 1):
            if torch.sum(s_index == i) > 0:
                psychometric[trial, i] = torch.mean(choice[s_index == i].float())

        loss[trial] = nn.MSELoss(reduction="sum")(outputs, target.to(device)).item()
        accuracy[trial] = torch.sum(choice == params["labels"].to(device)) / choice.size(0)
        evidence[trial] = torch.mean(choice_evidence)
        fixation[trial] = torch.mean(c_fixation)

    return loss, accuracy, evidence, fixation, psychometric


def evaluate_model(jobid, model_index, perturb_ratios, perturb_targets, num_trials, psychometric_edges, perturb_type):
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
        task_type=args["task_type"] if "task_type" in args else "embedded",
        mask=task_params["mask"] if "mask" in task_params else "none",
    )
    task.cursors = task_params["cursors"]

    net = models.build_model(args, task)
    net.load_state_dict(model)
    net = net.to(device)

    num_ratios = len(perturb_ratios)
    num_targets = len(perturb_targets)

    loss = torch.zeros(num_ratios, num_targets, num_trials)
    accuracy = torch.zeros(num_ratios, num_targets, num_trials)
    evidence = torch.zeros(num_ratios, num_targets, num_trials)
    fixation = torch.zeros(num_ratios, num_targets, num_trials)
    psychometric = torch.zeros(num_ratios, num_targets, num_trials, len(psychometric_edges) - 1)

    for i, perturb_ratio in enumerate(tqdm(perturb_ratios)):
        for it, target in enumerate(perturb_targets):
            cl, ca, ce, cf, cp = test_and_perturb(
                net,
                task,
                psychometric_edges,
                perturb_ratio=perturb_ratio,
                perturb_target=target,
                num_trials=num_trials,
                perturb_type=perturb_type,
            )
            loss[i, it] = cl
            accuracy[i, it] = ca
            evidence[i, it] = ce
            fixation[i, it] = cf
            psychometric[i, it] = cp

    results = dict(
        jobid=jobid,
        model_index=model_index,
        args=args,
        task=task_params,
        train_loss=train_loss,
        perturb_ratios=perturb_ratios,
        perturb_targets=perturb_targets,
        loss=loss,
        accuracy=accuracy,
        evidence=evidence,
        fixation=fixation,
        psychometric=psychometric,
    )

    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Perturbation analysis")
    parser.add_argument("--jobid", type=int, help="Job ID", default=7)
    parser.add_argument("--suffix", type=str, help="Suffix for model and results files", default="")
    parser.add_argument("--num_trials", type=int, help="Number of trials", default=100)
    parser.add_argument("--num_ratios", type=int, help="Number of perturbation ratios", default=11)
    parser.add_argument("--perturb_type", type=str, default="random")
    args = parser.parse_args()

    jobid = args.jobid
    suffix = args.suffix
    if suffix != "" and suffix[0] != "_":
        suffix = f"_{suffix}"
    directory = filepath / f"{jobid}"

    # Get all models (from the directory, of the form f"model_{i}.pt")
    model_indices = sorted([int(f.stem.split("_")[-1]) for f in directory.glob("model_*.pt")])

    # Set up perturbation analyses
    perturb_type = args.perturb_type
    if perturb_type == "random":
        perturb_ratios = torch.linspace(0, 1, args.num_ratios)
    elif perturb_type == "rotation":
        perturb_ratios = torch.linspace(0, torch.pi / 2, args.num_ratios)
    else:
        raise ValueError(f"Didn't recognize perturb type, received: {perturb_type}, allowed: ['random', 'rotation']")

    perturb_targets = ["gain", "tau", "threshold", "receptive", "projective"]
    psychometric_edges = create_cdf_proportional_bins(20, std_dev=1, range_multiplier=2)

    num_ratios = len(perturb_ratios)
    num_trials = args.num_trials
    num_targets = len(perturb_targets)
    num_models = len(model_indices)
    num_centers = len(psychometric_edges) - 1

    loss = torch.zeros(num_models, num_ratios, num_targets, num_trials)
    accuracy = torch.zeros(num_models, num_ratios, num_targets, num_trials)
    evidence = torch.zeros(num_models, num_ratios, num_targets, num_trials)
    fixation = torch.zeros(num_models, num_ratios, num_targets, num_trials)
    psychometric = torch.zeros(num_models, num_ratios, num_targets, num_centers)

    for i, model_index in enumerate(tqdm(model_indices)):
        results = evaluate_model(jobid, model_index, perturb_ratios, perturb_targets, num_trials, psychometric_edges, perturb_type)
        loss[i] = results["loss"]
        accuracy[i] = results["accuracy"]
        evidence[i] = results["evidence"]
        fixation[i] = results["fixation"]
        psychometric[i] = torch.mean(results["psychometric"], dim=2)

        # Save results every time it's finished so if the timeout happens we still have results...
        results = dict(
            jobid=jobid,
            model_indices=model_indices[:i],
            perturb_ratios=perturb_ratios,
            perturb_targets=perturb_targets,
            psychometric_edges=psychometric_edges,
            loss=loss[:i],
            accuracy=accuracy[:i],
            evidence=evidence[:i],
            fixation=fixation[:i],
            psychometric=psychometric[:i],
        )

        torch.save(results, directory / f"perturb_results_{perturb_type}{suffix}.pt")

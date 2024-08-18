import os
from pathlib import Path
from tqdm import tqdm

from argparse import ArgumentParser

import torch
import torch.nn as nn

import tasks
import models


def set_directory():
    job_path = Path("./jobs")
    if not job_path.exists():
        job_path.mkdir(parents=True, exist_ok=True)

    # check existing job folders and create a new one with the lowest number possible
    job_folders = [int(str(folder).split("/")[-1]) for folder in Path("./jobs").iterdir() if folder.is_dir()]
    if len(job_folders) == 0:
        directory = Path(f"./jobs/0")
    else:
        directory = Path(f"./jobs/{max(job_folders)+1}")
    print(f"This script is saving at: {directory}.")

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    return directory


def get_args():
    parser = ArgumentParser(description="Intrinsic excitability")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--network_type", type=str, default="Tau")
    parser.add_argument("--task_type", type=str, default="privileged")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--input_dimensions", type=int, default=10)
    parser.add_argument("--num_neurons", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--start_sigma", type=float, default=0.1)
    parser.add_argument("--end_sigma", type=float, default=0.1)
    parser.add_argument("--stim_time", type=int, default=20)
    parser.add_argument("--start_delay", type=int, default=5)
    parser.add_argument("--end_delay", type=int, default=30)
    parser.add_argument("--start_source_floor", type=float, default=0.5)
    parser.add_argument("--end_source_floor", type=float, default=0.5)
    parser.add_argument("--input_rank", type=int, default=3)
    parser.add_argument("--recurrent_rank", type=int, default=1)
    parser.add_argument("--nlfun", type=str, default="tanh")
    parser.add_argument("--gainfun", type=str, default="sigmoid")
    parser.add_argument("--taufun", type=str, default="sigmoid")
    parser.add_argument("--tauscale", type=int, default=10)
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--no_recurrent_learning", default=False, action="store_true")
    parser.add_argument("--no_intrinsic_learning", default=False, action="store_true")
    parser.add_argument("--no_input_learning", default=False, action="store_true")
    parser.add_argument("--no_readout_learning", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = get_args()
    directory = set_directory()

    # Number of models to train
    num_models = args.num_models

    # Hyperparameters
    B = args.batch_size  # Batch size
    D = args.input_dimensions  # Input dimensions
    learning_rate = args.learning_rate
    num_epochs = (args.num_epochs // 3) * 3

    start_sigma = args.start_sigma
    end_sigma = args.end_sigma
    sigma = torch.cat(
        (start_sigma * torch.ones(num_epochs // 3), torch.linspace(start_sigma, end_sigma, num_epochs // 3), end_sigma * torch.ones(num_epochs // 3))
    )

    start_delay = args.start_delay
    end_delay = args.end_delay
    delay_time = torch.cat(
        (
            start_delay * torch.ones(num_epochs // 3, dtype=torch.int),
            torch.linspace(start_delay, end_delay, num_epochs // 3, dtype=torch.int),
            end_delay * torch.ones(num_epochs // 3, dtype=torch.int),
        )
    )

    start_source_floor = args.start_source_floor
    end_source_floor = args.end_source_floor
    source_floor = torch.cat(
        (
            start_source_floor * torch.ones(num_epochs // 3),
            torch.linspace(start_source_floor, end_source_floor, num_epochs // 3),
            end_source_floor * torch.ones(num_epochs // 3),
        )
    )

    if args.task_type == "embedded":
        D = 2
        args.input_dimensions = 2

    task = tasks.ContextualGoNogo(D, sigma, stim_time=args.stim_time, delay_time=args.end_delay, num_contexts=2, task_type=args.task_type)

    def loss_function(outputs, targets, mask=None):
        difference = (outputs - targets).pow(2)
        if mask is None:
            return difference.mean()
        else:
            return (difference * mask.view(-1, 1).expand(-1, targets.size(2))).mean()

    for imodel in range(num_models):
        # Create network
        net = models.build_model(args, task)
        net = net.to(device)

        if args.no_input_learning:
            turn_off = ["input_weights", "input_receptive", "input_projective"]
            for name, prm in net.named_parameters():
                if name in turn_off:
                    prm.requires_grad = False

        if args.no_recurrent_learning:
            turn_off = ["recurrent_weights", "reccurent_receptive", "reccurent_projective"]
            for name, prm in net.named_parameters():
                if name in turn_off:
                    prm.requires_grad = False

        if args.no_intrinsic_learning:
            turn_off = ["hidden_gain", "hidden_threshold", "hidden_tau"]
            for name, prm in net.named_parameters():
                if name in turn_off:
                    prm.requires_grad = False

        if args.no_readout_learning:
            turn_off = ["readout"]
            for name, prm in net.named_parameters():
                if name in turn_off:
                    prm.requires_grad = False
        else:
            turn_off = ["readout_scale"]
            for name, prm in net.named_parameters():
                if name in turn_off:
                    prm.requires_grad = False

        # Save initial model
        torch.save(net.state_dict(), directory / f"init_model_{imodel}.pt")

        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 3, gamma=0.3)

        train_loss = torch.zeros(num_epochs)
        train_accuracy = torch.zeros(num_epochs)
        train_evidence = torch.zeros(num_epochs)
        train_fixation = torch.zeros(num_epochs)

        # Training loop
        for epoch in range(num_epochs):
            X, target, params = task.generate_data(
                B,
                sigma=sigma[epoch],
                delay_time=delay_time[epoch],
                source_strength=1.0,
                source_floor=source_floor[epoch],
                mask=args.mask,
            )

            optimizer.zero_grad()
            outputs, hidden = net(X.to(device), return_hidden=True)
            loss = loss_function(outputs, target.to(device), mask=args.mask if args.mask is None else params["loss_mask"].to(device))
            loss.backward()
            optimizer.step()

            choice, evidence, fixation = task.analyze_response(outputs, delay_time=delay_time[epoch], mask=args.mask)
            choice_evidence = evidence[:, 1] - evidence[:, 0]
            choice_evidence[params["labels"] == 0] *= -1

            train_loss[epoch] = loss.item()
            train_accuracy[epoch] = torch.sum(choice == params["labels"].to(device)) / choice.size(0)
            train_evidence[epoch] = torch.mean(choice_evidence)
            train_fixation[epoch] = torch.mean(fixation)

            if (epoch + 1) % max(num_epochs // 100, 1) == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy[epoch]:.4f}, Evidence: {train_evidence[epoch]:.4f}, Fixation: {train_fixation[epoch]:.4f}"
                )

        # Save the results
        results = dict(
            args=vars(args),
            task=vars(task),
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            train_evidence=train_evidence,
            train_fixation=train_fixation,
        )
        torch.save(results, directory / f"results_{imodel}.pt")

        # Save the network
        torch.save(net.state_dict(), directory / f"model_{imodel}.pt")

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

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        directory = Path(f"./jobs/{job_id}")
        print(f"The current Slurm job ID is: {job_id}, creating a new directory at {directory}")
    else:
        # check existing job folders and create a new one with the lowest number possible
        job_folders = [int(str(folder).split("/")[-1]) for folder in Path("./jobs").iterdir() if folder.is_dir()]
        if len(job_folders) == 0:
            directory = Path(f"./jobs/0")
        else:
            directory = Path(f"./jobs/{max(job_folders)+1}")
        print(f"This script is not running as a Slurm job or SLURM_JOB_ID is not set, saving at: {directory}.")

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    return directory


def get_args():
    parser = ArgumentParser(description="Intrinsic excitability")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--input_dimensions", type=int, default=10)
    parser.add_argument("--num_neurons", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--start_sigma", type=float, default=0.1)
    parser.add_argument("--end_sigma", type=float, default=0.5)
    parser.add_argument("--start_delay", type=int, default=1)
    parser.add_argument("--end_delay", type=int, default=10)
    parser.add_argument("--input_rank", type=int, default=3)
    parser.add_argument("--recurrent_rank", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = get_args()
    directory = set_directory()

    # Hyperparameters
    B = args.batch_size  # Batch size
    D = args.input_dimensions  # Input dimensions
    N = args.num_neurons  # Number of recurrent neurons
    learning_rate = args.learning_rate
    num_epochs = (args.num_epochs // 3) * 3
    input_rank = args.input_rank
    recurrent_rank = args.recurrent_rank

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
    task = tasks.GoNogo(D, sigma, delay_time=1)

    # Create network
    net = models.GainRNN(task.input_dimensionality(), N, task.output_dimensionality(), input_rank=input_rank, recurrent_rank=recurrent_rank)
    net = net.to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    train_loss = torch.zeros(num_epochs)

    # Training loop
    for epoch in range(num_epochs):
        X, target, params = task.generate_data(B, sigma=sigma[epoch], delay_time=delay_time[epoch], source_strength=1.0, source_floor=0.5)

        optimizer.zero_grad()
        outputs, hidden = net(X.to(device), return_hidden=True)
        loss = loss_function(outputs, target.to(device))
        loss.backward()
        optimizer.step()

        train_loss[epoch] = loss.item()

        if (epoch + 1) % max(num_epochs // 100, 1) == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the results
    results = dict(
        vars(args),
        train_loss=train_loss,
    )
    torch.save(results, directory / "results.pt")

    # Save the network
    torch.save(net.state_dict(), directory / "model.pt")

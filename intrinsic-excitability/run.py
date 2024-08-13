from tqdm import tqdm

import torch
import torch.nn as nn

import tasks
import models

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    B = 200 # Batch size
    D = 10 # Input dimensions
    N = 50 # Number of recurrent neurons
    learning_rate = 1e-2
    num_epochs = 2400

    start_sigma = 0.1
    end_sigma = 0.5
    sigma = torch.cat((
        start_sigma*torch.ones(num_epochs//3), 
        torch.linspace(start_sigma, end_sigma, num_epochs//3),
        end_sigma*torch.ones(num_epochs//3)
    ))

    start_delay = 1
    end_delay = 10
    delay_time = torch.cat((
        start_delay * torch.ones(num_epochs//3, dtype=torch.int),
        torch.linspace(start_delay, end_delay, num_epochs//3, dtype=torch.int),
        end_delay*torch.ones(num_epochs//3, dtype=torch.int)
    ))
    task = tasks.GoNogo(D, sigma, delay_time=1)

    # Create network
    net = models.GainRNN(task.input_dimensionality(), N, task.output_dimensionality(), input_rank=3, recurrent_rank=1)
    net = net.to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    save_accuracy = torch.zeros(num_epochs).to(device)

    # Training loop
    for epoch in range(num_epochs):
        X, target, params = task.generate_data(B, sigma=sigma[epoch], delay_time=delay_time[epoch], source_strength=1.0, source_floor=0.5)

        optimizer.zero_grad()
        outputs, hidden = net(X.to(device), return_hidden=True)
        loss = loss_function(outputs, target.to(device))
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % max(num_epochs // 100, 1) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
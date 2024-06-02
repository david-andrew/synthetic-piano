
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from matplotlib import pyplot as plt

import pdb

class RandomDataset(Dataset):
    def __init__(self, num_samples, N, M, mask_size):
        self.num_samples = num_samples
        self.N = N
        self.M = M
        self.mask_size = mask_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        A = torch.rand(self.N, self.M)
        B = torch.rand(self.N, self.M)
        B_masked = B.clone()
        B_masked[-self.mask_size:] = 0  # mask the last `mask_size` entries
        return A, B_masked, B



class SineWaveDataset(Dataset):
    def __init__(self, num_samples, N, M, mask_size, freq_range=(1, 5)):
        self.num_samples = num_samples
        self.N = N
        self.M = M
        self.mask_size = mask_size
        self.freq_range = freq_range

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        base_freq = np.random.uniform(*self.freq_range)
        t = np.linspace(0, 1, self.N)
        
        A = np.hstack([np.sin(2 * np.pi * (base_freq * (i + 1)) * t).reshape(-1, 1) for i in range(self.M)])
        B = np.hstack([np.cos(2 * np.pi * (base_freq * (i + 1)) * t).reshape(-1, 1) for i in range(self.M)])
        
        A = torch.tensor(A, dtype=torch.float32)
        B = torch.tensor(B, dtype=torch.float32)
        
        B_masked = B.clone()
        B_masked[-self.mask_size:] = 0  # mask the last `mask_size` entries
        
        return A, B_masked, B




class SimpleTransformer(nn.Module):
    def __init__(self, input_dim:int, nhead:int, num_encoder_layers:int, num_decoder_layers:int, seq_len:int):
        super(SimpleTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        self.fc = nn.Linear(input_dim, input_dim)

        # Positional embedding
        self.position_embedding = nn.Embedding(seq_len, input_dim)

    def forward(self, x):
        # x: [N, 2M] (concatenated A and B)

        # Add positional embeddings
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        x = x + self.position_embedding(positions)

        x = x.permute(1, 0, 2)  # Transformer expects [S, B, F]
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)
        return self.fc(x)


def train():
    # Example usage:
    mask_size = 8
    seq_length = 16
    num_channels = 100
    model = SimpleTransformer(input_dim=2*num_channels, nhead=2, num_encoder_layers=2, num_decoder_layers=2, seq_len=seq_length)
    # dataset = RandomDataset(num_samples=1000, N=seq_length, M=num_channels, mask_size=mask_size)
    dataset = SineWaveDataset(num_samples=1000, N=seq_length, M=num_channels, mask_size=mask_size)

    # put model on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Hyperparameters
    learning_rate = 0.001
    epochs = 1000
    epoch_length = 200
    batch_size = 128

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for i in range(epoch_length):
            for A, B_masked, B in dataloader:
                # Concatenate A and B_masked
                inputs = torch.cat([A, B_masked], dim=-1)

                # Put on GPU
                inputs = inputs.to(device)
                B = B.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs[:, -mask_size:, num_channels:], B[:, -mask_size:])
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}]: {i}, Loss: {loss.item():.4f}")
        pdb.set_trace()
        ...



if __name__ == '__main__':
    train()
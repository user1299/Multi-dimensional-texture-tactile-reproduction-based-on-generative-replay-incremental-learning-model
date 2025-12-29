import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import copy


class ReplayBuffer:

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def add(self, data):
        if len(self.buffer) < self.max_size:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
            self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []

        if len(self.buffer) < batch_size:
            indices = np.arange(len(self.buffer))
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        return [self.buffer[i] for i in indices]

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"ReplayBuffer(size={len(self)}, max_size={self.max_size})"


class LightweightAdapter(nn.Module):

    def __init__(self, d_model, adapter_dim=64, dropout=0.1):
        super(LightweightAdapter, self).__init__()
        self.down_proj = nn.Linear(d_model, adapter_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(adapter_dim, d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.down_proj.weight, nonlinearity='relu')
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x + residual


class SimpleGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, seq_len=200, output_dim=4):
        super(SimpleGenerator, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim * seq_len),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        latent = self.encoder(x_flat)
        output_flat = self.decoder(latent)
        output = output_flat.reshape(batch_size, self.seq_len, self.output_dim)
        return output

    def generate_samples(self, num_samples, device):
        self.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.seq_len, self.output_dim).to(device)
            generated = self.forward(noise)
        return generated


class GenerativeReplayTrainer:

    def __init__(self, base_model, config, device):
        self.base_model = base_model
        self.config = config
        self.device = device

        self.replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)

        self.generator = SimpleGenerator(
            input_dim=config.enc_in,
            hidden_dim=128,
            seq_len=config.seq_len,
            output_dim=config.c_out
        ).to(device)

        self.adapters = self._create_adapters()

        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.generator_lr if hasattr(config, 'generator_lr') else 1e-3
        )

        self.mse_loss = nn.MSELoss()

        self.history = {
            'generator_losses': [],
            'replay_samples_generated': 0
        }

    def _create_adapters(self):
        adapters = nn.ModuleDict()
        if hasattr(self.base_model, 'encoder'):
            if hasattr(self.base_model.encoder, 'mamba_layers'):
                for i, layer in enumerate(self.base_model.encoder.mamba_layers):
                    adapter_name = f'encoder_layer_{i}'
                    d_model = self.config.d_model if hasattr(self.config, 'd_model') else 128
                    adapters[adapter_name] = LightweightAdapter(d_model).to(self.device)
        return adapters

    def train_generator(self, data_loader, epochs=50, verbose=True):
        self.generator.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                generated = self.generator(batch_x)
                loss = self.mse_loss(generated, batch_y)
                self.generator_optimizer.zero_grad()
                loss.backward()
                self.generator_optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(data_loader)
            self.history['generator_losses'].append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Generator Training Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

    def add_to_buffer(self, x_data, y_data, max_samples=100):
        num_samples = min(len(x_data), max_samples)
        for i in range(num_samples):
            self.replay_buffer.add((x_data[i].cpu(), y_data[i].cpu()))
        print(f"Added {num_samples} samples to replay buffer, current size: {len(self.replay_buffer)}")

    def generate_replay_batch(self, batch_size):
        self.generator.eval()
        if len(self.replay_buffer) == 0:
            return None, None
        replay_samples = self.replay_buffer.sample(batch_size)
        if not replay_samples:
            return None, None
        x_samples = torch.stack([sample[0] for sample in replay_samples]).to(self.device)
        y_samples = torch.stack([sample[1] for sample in replay_samples]).to(self.device)
        with torch.no_grad():
            generated_x = self.generator(x_samples)
            generated_y = y_samples
        return generated_x, generated_y

    def apply_adapters(self, x):
        if len(self.adapters) == 0:
            return x
        return x

    def save_generator(self, save_path):
        torch.save({
            'generator_state': self.generator.state_dict(),
            'generator_optimizer_state': self.generator_optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }, save_path)
        print(f"Generator saved to: {save_path}")

    def load_generator(self, load_path):
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state'])
            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state'])
            self.history = checkpoint['history']
            print(f"Generator loaded from {load_path}")
            return True
        return False

    def get_replay_loss(self, model, batch_size=32, replay_weight=0.5):
        replay_x, replay_y = self.generate_replay_batch(batch_size)
        if replay_x is None:
            return 0.0
        with torch.no_grad():
            dec_inp = torch.zeros_like(replay_y[:, -self.config.label_len:, :]).float()
            dec_inp = torch.cat([replay_x, dec_inp], dim=1).float().to(self.device)
            base_outputs, _ = self.base_model(replay_x, None, dec_inp, None)
        current_outputs, _ = model(replay_x, None, dec_inp, None)
        replay_loss = self.mse_loss(current_outputs, base_outputs.detach())
        return replay_weight * replay_loss

    def get_status(self):
        return {
            'buffer_size': len(self.replay_buffer),
            'generator_trained_epochs': len(self.history['generator_losses']),
            'last_generator_loss': self.history['generator_losses'][-1] if self.history['generator_losses'] else None
        }
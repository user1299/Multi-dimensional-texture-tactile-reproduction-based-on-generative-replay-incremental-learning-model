import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

try:
    from BiMamba4TS import Model
    from generative_replay import GenerativeReplayTrainer
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure all module files are in the same directory")


def prepare_task_data(data_folder, seq_len, pred_len, device):
    import glob

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_folder}")

    all_data = []

    for csv_file in csv_files:
        try:
            data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            all_data.append(data)
        except Exception as e:
            print(f"Failed to read file {csv_file}: {e}")

    if not all_data:
        raise ValueError("No valid data loaded")

    combined_data = np.vstack(all_data)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    X, Y = [], []
    for i in range(len(scaled_data) - seq_len - pred_len):
        X.append(scaled_data[i:i + seq_len])
        Y.append(scaled_data[i + seq_len:i + seq_len + pred_len])

    if len(X) == 0:
        raise ValueError("Data too short for sliding window")

    X = np.array(X)
    Y = np.array(Y)

    X_tensor = torch.FloatTensor(X).to(device)
    Y_tensor = torch.FloatTensor(Y).to(device)

    print(f"Task data prepared: {X.shape} -> {Y.shape}")

    return X_tensor, Y_tensor, scaler


def load_pretrained_model(args, device):
    print("Loading pretrained model...")

    possible_paths = [
        os.path.join(args.checkpoints, 'checkpoint.pth'),
        os.path.join(args.checkpoints, 'best_model.pth'),
        os.path.join(args.results, 'model.pth'),
        os.path.join(args.results, 'best_model.pth'),
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        raise FileNotFoundError("No pretrained model found")

    print(f"Loading model from {model_path}")

    model = Model(args).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Weight loading error: {e}")
        print("Attempting partial loading...")

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Partial loading: {len(pretrained_dict)}/{len(model_dict)} parameters")

    model.eval()
    return model


class IncrementalLearningManager:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.incremental_dir = os.path.join(args.results, 'incremental')
        os.makedirs(self.incremental_dir, exist_ok=True)

        self.base_model = load_pretrained_model(args, device)

        self.replay_trainer = GenerativeReplayTrainer(self.base_model, args, device)

        self.current_model = self.base_model

        self._unfreeze_for_incremental()

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.current_model.parameters()),
            lr=args.incremental_lr
        )

        self.criterion = nn.MSELoss()

        self.history = {
            'tasks': [],
            'task_performance': [],
            'config': vars(args)
        }

    def _unfreeze_for_incremental(self):
        for param in self.current_model.parameters():
            param.requires_grad = False

        unfreeze_layers = ['head']

        for name, param in self.current_model.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.current_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.current_model.parameters())
        print(f"Incremental trainable parameters: {trainable_params:,}/{total_params:,} "
              f"({trainable_params / total_params * 100:.2f}%)")

    def train_task(self, task_id, data_folder, epochs=30):
        print(f"\n{'=' * 60}")
        print(f"Training Task {task_id}: {data_folder}")
        print(f"{'=' * 60}")

        X, Y, scaler = prepare_task_data(
            data_folder,
            self.args.seq_len,
            self.args.pred_len,
            self.device
        )

        dataset = TensorDataset(X, Y)
        data_loader = DataLoader(
            dataset,
            batch_size=min(self.args.batch_size, len(X)),
            shuffle=True
        )

        if task_id == 0:
            print("Training generator...")
            self.replay_trainer.train_generator(data_loader, epochs=30)

        print("Adding data to replay buffer...")
        self.replay_trainer.add_to_buffer(X[:50], Y[:50])

        self.current_model.train()
        task_losses = []

        for epoch in range(epochs):
            epoch_loss = 0

            for batch_x, batch_y in data_loader:
                dec_inp = torch.zeros_like(batch_y[:, -self.args.label_len:, :]).float()
                dec_inp = torch.cat([batch_x, dec_inp], dim=1).float().to(self.device)

                outputs, _ = self.current_model(batch_x, None, dec_inp, None)

                main_loss = self.criterion(outputs, batch_y)

                replay_loss = self.replay_trainer.get_replay_loss(
                    self.current_model,
                    batch_size=self.args.batch_size,
                    replay_weight=self.args.replay_weight
                )

                total_loss = main_loss + replay_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            avg_loss = epoch_loss / len(data_loader)
            task_losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

        performance = self.evaluate_task(X[:100], Y[:100])

        task_info = {
            'id': task_id,
            'data_folder': data_folder,
            'samples': len(X),
            'losses': task_losses,
            'performance': performance
        }

        self.history['tasks'].append(task_info)
        self.history['task_performance'].append(performance)

        self.save_checkpoint(task_id)

        print(f"Task {task_id} completed!")
        print(f"Performance: MSE={performance['mse']:.6f}, MAE={performance['mae']:.6f}")

        return performance

    def evaluate_task(self, X_test, Y_test):
        self.current_model.eval()

        with torch.no_grad():
            dec_inp = torch.zeros_like(Y_test[:, -self.args.label_len:, :]).float()
            dec_inp = torch.cat([X_test, dec_inp], dim=1).float().to(self.device)

            outputs, _ = self.current_model(X_test, None, dec_inp, None)

            mse = self.criterion(outputs, Y_test).item()
            mae = nn.L1Loss()(outputs, Y_test).item()

            outputs_np = outputs.cpu().numpy().flatten()
            targets_np = Y_test.cpu().numpy().flatten()

            if len(outputs_np) > 1:
                correlation = np.corrcoef(outputs_np, targets_np)[0, 1]
            else:
                correlation = 0.0

        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation
        }

    def save_checkpoint(self, task_id):
        checkpoint = {
            'task_id': task_id,
            'model_state': self.current_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'args': self.args
        }

        model_path = os.path.join(self.incremental_dir, f'model_task_{task_id}.pth')
        torch.save(checkpoint, model_path)

        generator_path = os.path.join(self.incremental_dir, f'generator_task_{task_id}.pth')
        self.replay_trainer.save_generator(generator_path)

        print(f"Checkpoint saved to: {model_path}")

    def load_checkpoint(self, task_id):
        model_path = os.path.join(self.incremental_dir, f'model_task_{task_id}.pth')

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)

            self.current_model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.history = checkpoint['history']

            print(f"Loaded checkpoint from task {task_id}")
            return True

        return False

    def save_results(self):
        history_path = os.path.join(self.incremental_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

        self.plot_performance()

        print(f"Results saved to: {self.incremental_dir}")

    def plot_performance(self):
        if not self.history['task_performance']:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        tasks = range(len(self.history['task_performance']))

        axes[0].plot(tasks, [p['mse'] for p in self.history['task_performance']], 'o-', linewidth=2)
        axes[0].set_xlabel('Task ID')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('MSE by Task')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(tasks, [p['mae'] for p in self.history['task_performance']], 's-', linewidth=2, color='orange')
        axes[1].set_xlabel('Task ID')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('MAE by Task')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(tasks, [p['correlation'] for p in self.history['task_performance']], '^-', linewidth=2,
                     color='green')
        axes[2].set_xlabel('Task ID')
        axes[2].set_ylabel('Correlation')
        axes[2].set_title('Correlation by Task')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.incremental_dir, 'performance_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='BiMamba4TS Incremental Learning')

    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')

    parser.add_argument('--seq_len', type=int, default=200, help='Sequence length')
    parser.add_argument('--pred_len', type=int, default=200, help='Prediction length')
    parser.add_argument('--label_len', type=int, default=100, help='Label length')
    parser.add_argument('--enc_in', type=int, default=4, help='Input dimension')
    parser.add_argument('--c_out', type=int, default=4, help='Output dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument('--replay_buffer_size', type=int, default=500, help='Replay buffer size')
    parser.add_argument('--replay_weight', type=float, default=0.3, help='Replay loss weight')
    parser.add_argument('--incremental_lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--incremental_epochs', type=int, default=30, help='Training epochs')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='Checkpoints directory')
    parser.add_argument('--results', type=str, default='./results', help='Results directory')
    parser.add_argument('--data_folders', type=str, nargs='+',
                        help='Data folder list (in order)')

    parser.add_argument('--resume_task', type=int, default=-1,
                        help='Resume from which task (-1 for start from beginning)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    manager = IncrementalLearningManager(args, device)

    if args.resume_task >= 0:
        success = manager.load_checkpoint(args.resume_task)
        if not success:
            print(f"Cannot resume from task {args.resume_task}, starting from beginning")
            args.resume_task = -1

    if args.data_folders:
        start_task = args.resume_task + 1 if args.resume_task >= 0 else 0
        total_tasks = len(args.data_folders)

        print(f"\nStarting incremental learning, total {total_tasks} tasks")
        print(f"Starting from task {start_task}")

        start_time = time.time()

        for task_id, data_folder in enumerate(args.data_folders[start_task:], start=start_task):
            if os.path.exists(data_folder):
                try:
                    performance = manager.train_task(
                        task_id,
                        data_folder,
                        epochs=args.incremental_epochs
                    )

                    print(f"Task {task_id} completed: MSE={performance['mse']:.6f}")

                except Exception as e:
                    print(f"Task {task_id} failed: {e}")
                    continue
            else:
                print(f"Skipping non-existent folder: {data_folder}")

        manager.save_results()

        print(f"\nIncremental learning completed! Total time: {time.time() - start_time:.2f} seconds")

    else:
        print("Please provide data folder paths (--data_folders)")
        print("Example: --data_folders /path/to/task1 /path/to/task2 /path/to/task3")


if __name__ == "__main__":
    main()
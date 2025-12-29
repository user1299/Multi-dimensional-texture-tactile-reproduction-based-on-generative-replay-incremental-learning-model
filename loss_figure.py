import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = "/SATA2/DY/bimamba2/results/loss.csv"
save_path = "/SATA2/DY/bimamba2/loss_curve.png"

loss_data = pd.read_csv(csv_file_path)

plt.figure(figsize=(10, 6))
plt.plot(loss_data['epoch'], loss_data['train_loss'], label='Training Loss', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)

plt.savefig(save_path)
plt.show()

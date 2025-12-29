import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')

file_path = r'/SATA2/DY/bimamba2/datasave/epoch2256/min_loss_batch.csv'
data = pd.read_csv(file_path)

outputs = data['outputs'][:200]
batch_y = data['batch_y'][:200]

plt.figure(figsize=(10, 6))
plt.plot(outputs, label='Real')
plt.plot(batch_y, label='Predict')
plt.xlabel('Time')
plt.ylabel('Accel')
plt.title('Comparison of Real and Predict')
plt.legend()
plt.grid(True)

save_path = r'/SATA2/DY/bimamba2/predictions/comparison_plot.png'

plt.savefig(save_path)
print(f"图像已保存至 {save_path}，请检查文件。")


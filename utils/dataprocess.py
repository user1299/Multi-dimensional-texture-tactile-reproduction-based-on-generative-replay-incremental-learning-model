import pandas as pd
import os
from tqdm import tqdm

speed_folder = r"E:/Learning materials/Project1/data/original_data/speed"
force_folder = r"E:/Learning materials/Project1/data/original_data/force"
accel_folder = r"E:/Learning materials/Project1/data/original_data/accel"
output_folder = r"C:/Users/dying/Desktop/data/hatt"
processed_folder = r"C:/Users/dying/Desktop/data/hatt_process"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)


def get_all_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.xlsx')]
    file_dict = {f.split('_', 1)[1]: os.path.join(folder, f) for f in files}
    return file_dict


speed_files = get_all_files(speed_folder)
force_files = get_all_files(force_folder)
accel_files = get_all_files(accel_folder)

for file_key, speed_file in tqdm(speed_files.items(), desc="处理文件", unit="file"):
    force_file = force_files.get(file_key)
    accel_file = accel_files.get(file_key)

    if force_file and accel_file:
        speed_data = pd.read_excel(speed_file)
        force_data = pd.read_excel(force_file)
        accel_data = pd.read_excel(accel_file)

        combined_df = pd.DataFrame({
            "speed": speed_data.iloc[:, 0].values,
            "force": force_data.iloc[:, 0].values,
            "accel": accel_data.iloc[:, 0].values
        })

        output_file_path = os.path.join(output_folder, f"{file_key}")

        if output_file_path.lower().endswith('.xlsx'):
            output_file_path = output_file_path[:-5]

        output_file_path += '.csv'

        combined_df.to_csv(output_file_path, index=False)
    else:
        print(f"缺少对应的force或accel文件: {file_key}")

for file in tqdm(os.listdir(output_folder), desc="处理hatt文件夹", unit="file"):
    if file.endswith('.csv'):
        file_path = os.path.join(output_folder, file)
        df = pd.read_csv(file_path)

        df['speed'] = df['speed'].iloc[200:].reset_index(drop=True)
        df['force'] = df['force'].iloc[200:].reset_index(drop=True)
        df['accel'] = df['accel'].iloc[:-200].reset_index(drop=True)

        min_len = min(len(df['speed']), len(df['force']), len(df['accel']))
        df = df.iloc[:min_len]

        processed_file_path = os.path.join(processed_folder, file)

        if processed_file_path.lower().endswith('.xlsx'):
            processed_file_path = processed_file_path[:-5]

        df.to_csv(processed_file_path, index=False)

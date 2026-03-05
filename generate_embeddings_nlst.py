from sybil import Serie, Sybil
import pandas as pd
from glob import glob
import os
import pyreadstat
from tqdm import tqdm
import torch
from safetensors.torch import save_file, load_file


# Load a trained model
model_str = "sybil_1"
model = Sybil(model_str)

# load pid2split
df = pd.read_csv("pid2split.csv")
pids = df['PID'].tolist()


root_img_dir = "/mii/data/lung/nlst/NLST_CT_raw/data"
save_dir = f"/hsuraid/avepa/nlst_{model_str}_embeddings"
validate_embeddings = True

print(f"Saving embeddings for {len(pids)} patients.")

for pid in tqdm(pids):
    patient_dir = os.path.join(root_img_dir, str(pid))
    if os.path.exists(patient_dir):
        time_points = sorted(glob(os.path.join(patient_dir, "*")))
        for time_index, time_point in enumerate(time_points):
            img_dirs = sorted(glob(os.path.join(time_point, "*")))
            min_img_files = None
            for img_dir_pos, img_dir in enumerate(img_dirs):
                img_files = glob(os.path.join(img_dir, "*"))
                # skip localizers with less than 20 slices
                if len(img_files) < 20:
                    continue
                if min_img_files is None or len(img_files) < len(min_img_files):
                    min_img_files = img_files
                    min_img_dir_pos = img_dir_pos
            if min_img_files is None or len(min_img_files) < 20:
                continue
            save_path = os.path.join(save_dir, f"pid{pid}_ts{time_index}.st")
            if os.path.exists(save_path):
                continue
            # Create a Serie object for the patient
            try:
                serie = Serie(min_img_files)
            except:
                continue
            volume = serie.get_volume()
            volume = volume.to(model.device)
            embeddings = []
            for model_ in model.ensemble:
                with torch.no_grad():
                    embeddings_ = model_.image_encoder(volume)
                    embeddings.append(embeddings_)
            embeddings = torch.cat(embeddings, dim=0)
            if list(embeddings.size())[1:] != [512, 25, 16, 16]:
                continue
            if embeddings.size()[0] == 1:
                embeddings = embeddings.squeeze(0)
                print("No averaging")
            else:
                embeddings = torch.mean(embeddings, dim=0)
            embeddings_file_path = os.path.join(save_dir, f"pid{pid}_ts{time_index}.st")
            if validate_embeddings:
                loaded_embeddings = load_file(embeddings_file_path)['embeddings']
                if not torch.allclose(embeddings, loaded_embeddings):
                    print(f"Validation failed for {embeddings_file_path}")
            else:
                save_file({"embeddings": embeddings}, embeddings_file_path) 
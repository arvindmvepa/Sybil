from sybil import Serie, Sybil
import pandas as pd
from glob import glob
import os
import pyreadstat
from tqdm import tqdm
import torch
from safetensors.torch import save_file


# Load a trained model
model = Sybil("sybil_ensemble")

# load pid2split
df = pd.read_csv("pid2split.csv")
pids = df['PID'].tolist()


root_img_dir = "/mii/data/lung/nlst/NLST_CT_raw/data"
save_dir = "/hsuraid/avepa/nlst_sybil_embeddings"

print(f"Saving embeddings for {len(pids)} patients.")

for pid in tqdm(pids[:7500]):
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
            # Create a Serie object for the patient
            serie = Serie(min_img_files)
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
            embeddings = torch.mean(embeddings, dim=0)
            save_file({"embeddings": embeddings}, os.path.join(save_dir, f"pid{pid}_ts{time_index}.st"))  
            


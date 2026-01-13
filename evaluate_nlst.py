from sybil import Serie, Sybil
import pandas as pd
from glob import glob
import os


# Load a trained model
model = Sybil("sybil_ensemble")

# load pid2split
df = pd.read_csv("pid2split.csv")
test_df = df[df['SPLIT'] == 'test']
test_pids = test_df['PID'].tolist()


img_dir = "/mii/data/lung/nlst/NLST_CT_raw/data"
print(f"Evaluating {len(test_pids)} test patients.")
for test_pid in test_pids:
    patient_dir = os.path.join(img_dir, str(test_pid))
    time_points = sorted(glob(os.path.join(patient_dir, "*")))
    print(f"Evaluating Patient ID: {test_pid} with {len(time_points)} time points.")
    for time_index, time_point in enumerate(time_points):
        img_dirs = sorted(glob(os.path.join(time_point, "*")))
        max_img_files = None
        for img_dir_pos, img_dir in enumerate(img_dirs):
            img_files = glob(os.path.join(img_dir, "*"))
            if max_img_files is None or len(img_files) > len(max_img_files):
                max_img_files = img_files
                max_img_dir_pos = img_dir_pos
        # Create a Serie object for the patient
        serie = Serie(max_img_files)
        # Get risk score
        score = model.predict([serie])
        print(f"Patient ID: {test_pid}, Time Point: {time_index}, Risk Score: {score}")
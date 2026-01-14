from sybil import Serie, Sybil
import pandas as pd
from glob import glob
import os
import pyreadstat
from tqdm import tqdm


# Load a trained model
model = Sybil("sybil_ensemble")

# load pid2split
df = pd.read_csv("pid2split.csv")
test_df = df[df['SPLIT'] == 'test']
test_pids = test_df['PID'].tolist()


root_img_dir = "/mii/data/lung/nlst/NLST_CT_raw/data"
patient_file = "participant_d100814.sas7bdat"
(patient_df, _) = pyreadstat.read_sas7bdat(patient_file)

gt_labels = []
preds = []
print(f"Evaluating {len(test_pids)} test patients.")
for test_pid in tqdm(test_pids):
    pid_ann_df = patient_df.loc[patient_df["pid"] == test_pid]
    cancyr = pid_ann_df['cancyr'].iloc[0]
    has_cancer = 0
    if not pd.isna(cancyr):
        has_cancer = 1
    
    patient_dir = os.path.join(root_img_dir, str(test_pid))
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
            try:
                # Create a Serie object for the patient
                serie = Serie(min_img_files)
                # Get risk score
                pred = model.predict([serie])
                gt_labels.append(has_cancer)
                preds.append(pred.scores[0][-1])
                #print(f"Patient ID: {test_pid}, Time Point: {time_index}, Risk Score: {pred.scores}, {pred.scores[0][-1]}")
            except Exception as e:
                print(f"Error processing patient {test_pid}, time point {time_index}: {str(e)}")

# Calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(gt_labels, preds)
print(f"AUC: {auc}")

# Calculate accuracy at threshold 0.5
pred_labels = [1 if p >= 0.5 else 0 for p in preds]
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(gt_labels, pred_labels)
print(f"Accuracy: {accuracy}")

# Calculate sensitivity and specificity
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(gt_labels, pred_labels).ravel().tolist()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
import csv
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy import stats


class Patient: 
    all_patients = []

    def __init__(self, DonorID, sex=None, death_age=None, cog_stat=None, 
                 consensus_dx=None, brain_weight=None, mmse_score=None):
        self.DonorID = DonorID
        self.sex = sex
        self.death_age = death_age
        self.cog_stat = cog_stat
        self.consensus_dx = consensus_dx  # list of diagnoses or None
        self.brain_weight = brain_weight
        self.mmse_score = mmse_score      # last MMSE score (integer)
        Patient.all_patients.append(self)

    def __repr__(self):
        dx_display = self.consensus_dx if self.consensus_dx else "None"
        return (f"{self.DonorID} | sex: {self.sex} | Death Age {self.death_age} | "
                f"Cognitive Status {self.cog_stat} | Consensus Dx {dx_display} | "
                f"Brain Wt {self.brain_weight} | Last MMSE: {self.mmse_score}")

    @classmethod
    def instantiate_from_csv(cls, filename: str):
        with open(filename, encoding="utf8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            for row in reader:
                donor_id = row["Donor ID"].strip()
                sex = row["Sex"].strip() if row["Sex"] else None
                death_age = int(row["Age at Death"]) if row["Age at Death"] else None
                cog_stat = row["Cognitive Status"].strip() if row["Cognitive Status"] else None

                # collect diagnoses marked "Checked"
                consensus_cols = [h for h in headers if h.startswith("Consensus Clinical Dx")]
                diagnoses = []
                for col in consensus_cols:
                    val = row[col].strip()
                    if val == "Checked":
                        dx_name = col.replace("Consensus Clinical Dx (choice=", "").replace(")", "")
                        diagnoses.append(dx_name)
                if not diagnoses:
                    diagnoses = None

                # brain weight
                brain_weight = None
                if row["Fresh Brain Weight"]:
                    try:
                        brain_weight = float(row["Fresh Brain Weight"])
                    except ValueError:
                        brain_weight = None

                # last MMSE score
                mmse_score = None
                if "Last MMSE Score" in row and row["Last MMSE Score"]:
                    try:
                        mmse_score = int(row["Last MMSE Score"])
                    except ValueError:
                        mmse_score = None

                # create patient object
                Patient(
                    DonorID=donor_id,
                    sex=sex,
                    death_age=death_age,
                    cog_stat=cog_stat,
                    consensus_dx=diagnoses,
                    brain_weight=brain_weight,
                    mmse_score=mmse_score
                )

# ----------------------------
# Load Patients
# ----------------------------
Patient.instantiate_from_csv("UpdatedMetaData.csv")
Patient.all_patients.sort(key=lambda p: p.consensus_dx or [], reverse=False)

# ----------------------------
# Count No dementia vs Dementia
# ----------------------------
print("\nTotal Patients with Dementia:")
no_dementia_count = sum(1 for p in Patient.all_patients if p.cog_stat == "No dementia")
dementia_count = sum(1 for p in Patient.all_patients if p.cog_stat != "No dementia")
total_patients = len(Patient.all_patients)

print(f"\nNumber of 'No dementia' patients: {no_dementia_count}")
print(f"Number of 'Dementia/Other' patients: {dementia_count}")
print(f"Total patients: {total_patients}")



# ----------------------------
# Count each individual consensus diagnosis
# ----------------------------
from collections import Counter

dx_counter = Counter()

for p in Patient.all_patients:
    if p.consensus_dx:  # skip if None
        dx_counter.update(p.consensus_dx)

print("\nConsensus diagnosis counts:")
for dx, count in dx_counter.items():
    print(f"{dx}: {count}")



# ----------------------------
# Bar chart: Alzheimer’s only vs Alzheimer’s+Other vs Control
# ----------------------------
group_counts = {"Alzheimer’s only": 0, "Alzheimer’s + Other": 0, "Control": 0}

for p in Patient.all_patients:
    if not p.consensus_dx:
        continue

    dx_list = p.consensus_dx

    # Control
    if "Control" in dx_list:
        group_counts["Control"] += 1

    # Alzheimer’s categories
    elif any(dx in ["Alzheimers disease", "Alzheimers Possible/ Probable"] for dx in dx_list):
        if len(dx_list) == 1:  # only AD
            group_counts["Alzheimer’s only"] += 1
        else:  # AD plus at least one other
            group_counts["Alzheimer’s + Other"] += 1

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(group_counts.keys(), group_counts.values(), 
        color=["skyblue", "lightcoral", "lightgreen"])
plt.ylabel("Number of Patients")
plt.title("Patient Counts: Alzheimer’s Only vs Alzheimer’s+Other vs Control")

# add counts on bars
for i, v in enumerate(group_counts.values()):
    plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
# ----------------------------
# ANOVA: Compare Last MMSE across 3 groups
# ----------------------------
alz_only_scores = []
alz_plus_other_scores = []
control_scores = []

for p in Patient.all_patients:
    if p.mmse_score is None or not p.consensus_dx:
        continue
    dx_list = p.consensus_dx

    # Control
    if "Control" in dx_list:
        control_scores.append(p.mmse_score)

    # Alzheimer’s categories
    elif any(dx in ["Alzheimers disease", "Alzheimers Possible/ Probable"] for dx in dx_list):
        if len(dx_list) == 1:  # only AD
            alz_only_scores.append(p.mmse_score)
        else:  # AD + something else
            alz_plus_other_scores.append(p.mmse_score)

print("\nGroup sizes and means (Last MMSE):")
print(f"Alzheimer’s only: n={len(alz_only_scores)}, mean={np.mean(alz_only_scores) if alz_only_scores else 'N/A'}")
print(f"Alzheimer’s + Other: n={len(alz_plus_other_scores)}, mean={np.mean(alz_plus_other_scores) if alz_plus_other_scores else 'N/A'}")
print(f"Control: n={len(control_scores)}, mean={np.mean(control_scores) if control_scores else 'N/A'}")

if alz_only_scores and alz_plus_other_scores and control_scores:
    f_stat, p_value = stats.f_oneway(alz_only_scores, alz_plus_other_scores, control_scores)
    print("ANOVA F-statistic:", f_stat)
    print("ANOVA p-value:", p_value)

    plt.figure(figsize=(8, 6))
    plt.boxplot([alz_only_scores, alz_plus_other_scores, control_scores],
                tick_labels=["Alzheimer’s only", "Alzheimer’s + Other", "Control"],
                patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='blue'),
                medianprops=dict(color='red'))
    plt.title("Last MMSE Score Across Groups\nANOVA p-value = {:.4f}".format(p_value))
    plt.ylabel("Last MMSE Score")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Not enough data in one or more groups to run ANOVA.")
#9) GET PATIENT ATTRIBUTES THAT WE WANT TO COMPARE ON A SCATTER PLOT

death_age_list = []
mmse_scores = []

for patient in Patient.all_patients:
    if patient.death_age is not None and patient.mmse_score is not None:
        death_age_list.append(patient.death_age)
        mmse_scores.append(patient.mmse_score)

X = death_age_list  # Independent variable
y = mmse_scores     # Dependent variable

print("Ages at death:", X)
print("MMSE scores:", y)

#10) VISUALIZE DATA ON A SCATTER PLOT

plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.7)
plt.xlabel('Age at Death')
plt.ylabel('Last MMSE Score')
plt.title('Scatter Plot of Age at Death vs Last MMSE Score')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#11) EXPORT DATA TO A .csv FILE

import pandas as pd

df = pd.DataFrame({
    'Age at Death': X,
    'Last MMSE Score': y
})

df.to_csv('patient_data.csv', index=False)
print("CSV file 'patient_data.csv' has been created.")

# ----------------------------
# Linear Regression: Age of Death vs Last MMSE
# ----------------------------
from sklearn.linear_model import LinearRegression

# Clean data (remove None values)
clean_data = [(p.death_age, p.mmse_score) for p in Patient.all_patients 
              if p.death_age is not None and p.mmse_score is not None]

if clean_data:
    X, y = zip(*clean_data)
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    # Plot scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label="Data", alpha=0.7)

    # Plot regression line
    x_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    plt.plot(x_range, model.predict(x_range), color="red", label="Regression Line")

    # Annotate equation
    equation = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.2f}"
    plt.text(min(X), max(y), equation, color="red", fontsize=12, verticalalignment='top')

    plt.xlabel("Age of Death")
    plt.ylabel("Last MMSE Score")
    plt.title("Age of Death vs Last MMSE Score (with Regression Line)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No valid data for regression (missing Age or MMSE).")


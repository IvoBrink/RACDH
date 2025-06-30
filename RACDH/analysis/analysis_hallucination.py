import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency, norm
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
import warnings

# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.print import *
from RACDH.config import params

if __name__ == "__main__":
    datasets = ["squad", "squad-switch"]
    classifier = "logreg"
    models = ["Llama-3.1-8B", "Mistral-7B-v0.1"]
    for dataset in datasets:
        for model in models:
            file_tag = "normal" if dataset == "squad" else "parametric"
            samples = load_json(f"plots/{model}/results_squad_{classifier}__{file_tag}.json")
            print(f"\nModel: {model} | Dataset: {dataset}")
            df = pd.DataFrame(samples)

            # binary flags
            df["correct"]   = df["answer_correct"].astype(bool)
            df["gold"]      = "Contextual" if dataset == "squad" else "Parametric" 
            df["mismatch"]  = (df["label"] != df["gold"])  # True - wrong channel

            # 2 × 2 contingency: rows = mismatch / no-mismatch, cols = correct / incorrect
            a = ((~df["mismatch"]) &  df["correct"]).sum()   # no-mismatch & correct
            b = ((~df["mismatch"]) & ~df["correct"]).sum()   # no-mismatch & incorrect
            c = ( df["mismatch"]  &  df["correct"]).sum()    #    mismatch & correct
            d = ( df["mismatch"]  & ~df["correct"]).sum()    #    mismatch & incorrect

            table = np.array([[a, b],
                              [c, d]])

            print("Contingency table (rows=mismatch, cols=correctness)")
            print(pd.DataFrame(table,
                index=["no mismatch","mismatch"],
                columns=["correct","incorrect"]))
            print()

            # Fisher’s exact test (two-sided) – odds ratio & p-value
            odds_ratio, p_fisher = fisher_exact(table, alternative='two-sided')
            print(f"Fisher exact odds-ratio = {odds_ratio:0.3f},  p = {p_fisher:0.3g}")

            # Chi test of independence
            chi2, p_chi, dof, expected = chi2_contingency(table, correction=False)
            print(f"χ²({dof}) = {chi2:0.2f},  p = {p_chi:0.3g}")

            # Relative risk  + 95 % Wald CI   (risk = P(incorrect))
            risk_mismatch     = d / (c + d)
            risk_no_mismatch  = b / (a + b)
            rel_risk          = risk_mismatch / risk_no_mismatch

            # Wald SE on log(RR)
            se_log_rr = np.sqrt(1/b - 1/(a+b) + 1/d - 1/(c+d))
            z = norm.ppf(0.975)
            ci_low, ci_high = (
                np.exp(np.log(rel_risk) - z * se_log_rr),
                np.exp(np.log(rel_risk) + z * se_log_rr)
            )

            print(f"Relative risk = {rel_risk:0.3f}  "
                  f"(95 % CI {ci_low:0.3f} – {ci_high:0.3f})")

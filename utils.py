import os
from os import path
import torch
import pandas as pd
import numpy as np

from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score, 
    f1_score,
    precision_recall_curve,
    auc
)

class CSVLogger_my:
    def __init__(self, columns, file) :
        self.columns=columns
        self.file=file
        if not self.check_header():
            self._write_header()

    def check_header(self):
        if path.exists(self.file):
            header=True
        else:
            header=False
        return header

    def _write_header(self):
        with open(self.file,"a") as f:
            string=""
            for attrib in self.columns:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

    def log(self,row) :
        if len(row)!=len(self.columns) :
            raise Exception("Mismatch between row vector and number of columns in logger")
        with open(self.file, "a") as f:
            string=""
            for attrib in row:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
import math
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix
)

def score(y_true, y_pred):

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    else:
        y_true = np.asarray(y_true)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = np.asarray(y_pred)

    y_true = y_true.reshape(-1).astype(int)
    y_pred = y_pred.reshape(-1).astype(float)

    auc_roc = roc_auc_score(y_true, y_pred)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    prauc = auc(recall, precision)

    y_pred_bin = np.round(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    acc = (tp + tn) / (tp + fn + tn + fp) if (tp + fn + tn + fp) > 0 else 0.0

    denom = (tp + fn) * (tp + fp) * (tn + fn) * (tn + fp)
    mcc = ((tp * tn - fn * fp) / math.sqrt(denom)) if denom > 0 else 0.0

    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * P * se) / (P + se) if (P + se) > 0 else 0.0

    ba = (se + sp) / 2.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        'tp': tp,
        'tn': tn,
        'fn': fn,
        'fp': fp,
        'sensitivity': se,
        'specificity': sp,
        'mcc': mcc,
        'accuracy': acc,
        'roc_auc': auc_roc,
        'f1_score': f1,
        'balanced_accuracy': ba,
        'pr_auc': prauc,
        'ppv': ppv,
        'npv': npv
    }


def metrics_from_log(log_path):
  df = pd.read_csv(log_path)
  metrics = {
      "tp": np.max(df['tp']),
      "tn": np.max(df['tn']),
      "fn": np.max(df['fn']),
      "fp": np.max(df['fp']),
      "se": np.max(df['se']),
      "sp": np.max(df['sp']),
      "mcc": np.max(df['mcc']),
      "acc": np.max(df['acc']),
      "test_acc": np.max(df['auc_roc_score']),
      "f1": np.max(df['F1']),
      "ba": np.max(df['BA']),
      "prauc": np.max(df['prauc']),
      "ppv": np.max(df['PPV']),
      "npv": np.max(df['NPV']),
  }

  metrics_mean = {
      "tp": np.mean(df['tp']),
      "tn": np.mean(df['tn']),
      "fn": np.mean(df['fn']),
      "fp": np.mean(df['fp']),
      "se": np.mean(df['se']),
      "sp": np.mean(df['sp']),
      "mcc": np.mean(df['mcc']),
      "acc": np.mean(df['acc']),
      "test_acc": np.mean(df['auc_roc_score']),
      "f1": np.mean(df['F1']),
      "ba": np.mean(df['BA']),
      "prauc": np.mean(df['prauc']),
      "ppv": np.mean(df['PPV']),
      "npv": np.mean(df['NPV']),
  }
  return metrics, metrics_mean


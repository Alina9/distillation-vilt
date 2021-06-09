from sklearn.metrics import roc_auc_score, accuracy_score
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics.classification import AUROC
import torch.nn as nn
import wandb


def classification_metrics(y_true, y_score):
    #auroc = AUROC()
    softmax = nn.Softmax(dim=1)
    y_score = softmax(y_score)
    y_true = y_true.to(y_score.device)

    # accuracy
    y_pred = y_score.max(1)[1]
    acc = accuracy(y_true, y_pred)

    # roc auc
    y_score = y_score.detach().cpu().numpy()
    roc_auc = roc_auc_score(y_true.cpu(), y_score[:, 1])

    # roc_auc = auroc(y_true, y_pred)
    return acc.detach().item(), roc_auc

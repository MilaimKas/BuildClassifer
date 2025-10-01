from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve

def precision_at_min_recall(y_true, y_pred, min_recall=0.5):
    """
    Give the precision (on negative class) given a fix a minimum acceptable recall (precision for positive class).
    Use for model optimization if false positive is important

    Args:
        y_true (int): true flag.
        y_pred (float): predicted probability
        min_recall (float, optional): minimum desired recall (precision for positive class). Defaults to 0.8.

    Returns:
        float: precision_at_recall
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    try:
        valid = recall >= min_recall
        return precision[valid].max()
    except:
        return 0.0  # If no threshold satisfies the constraint

def partial_auc_fpr(y_true, y_pred, max_fpr=0.05):
    """
    Area under the precision-recall curve, or ROC curve, but only over the region of interest. 
    Which corresponds to the desire maximum false positive rate.

    Args:
        y_true (int): true labels.
        y_pred (float): predicted proba.
        max_fpr (float, optional): desired maximum fpr. Defaults to 0.05 (5%).

    Returns:
        float: bounded roc_auc value
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    mask = fpr <= max_fpr
    return auc(fpr[mask], tpr[mask])

def cost_sensitive_score(y_true, y_pred, fp_cost=10, fn_cost=1):
    """
    Returns a cost function using a fpr Vs fn weight.
    Examplefor default values:
        "blocking a good user = 10x worse than missing a scammer"

    Args:
        y_true (int): true labels.
        y_pred (float): predicted proba.
        fp_cost (float, optional): weigth on the fpr. Defaults to 10.
        fn_cost (float, optional): weight on the fn. Defaults to 1.

    Returns:
        float: cost value
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return -(fp * fp_cost + fn * fn_cost)
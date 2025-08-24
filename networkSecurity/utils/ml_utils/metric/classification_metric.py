from networkSecurity.entity.artifact_entity import ClassificationMetricArtifact
from networkSecurity.exceptionHandling.exception import NetworkSecurityException
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

def get_classification_metrics(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        return ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
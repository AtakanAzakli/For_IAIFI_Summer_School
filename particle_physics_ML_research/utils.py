import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

def compute_precision_recall(model, X_test, y_test, cfg):
    predictions = model.predict(X_test, batch_size=1024)
    precision = precision_score(y_test, predictions.round())
    recall = recall_score(y_test, predictions.round())
    if cfg.experiment.wandb:
        wandb.log({"precision": precision, "recall": recall})
    return precision, recall

def plot_roc_curve(model, X_test, y_test, cfg):
    predictions = model.predict(X_test, batch_size=1024)
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    roc_fig = plt.gcf()
    if cfg.experiment.wandb:
        wandb.log({"roc_curve": wandb.Image(roc_fig)})
    plt.show()
    plt.clf()

def plot_training_loss(history, cfg):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_fig = plt.gcf()
    if cfg.experiment.wandb:
        wandb.log({"training_validation_loss": wandb.Image(loss_fig)})
    plt.show()
    plt.clf()

def plot_confusion_matrix(model, X_test, y_test, cfg):
    predictions = model.predict(X_test, batch_size=1024)
    cm = confusion_matrix(y_test, predictions.round())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    conf_mat_fig = plt.gcf()
    if cfg.experiment.wandb:
        wandb.log({"confusion_matrix": wandb.Image(conf_mat_fig)})
    plt.show()
    plt.clf()

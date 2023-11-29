import streamlit as st

import matplotlib.pyplot as plt

from sklearn import metrics

class Results():
    def __init__(self, y_true, y_pred, fpr, tpr, auc_roc) -> None:
        self.y_true = y_true
        self.y_pred = y_pred
        self.fpr = fpr
        self.tpr = tpr
        self.auc_roc = auc_roc

    def plot_res(self):
        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.fpr, self.tpr, label=f'AUC = {self.auc_roc:.4f}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)

        # true v predicted labels
        st.divider()
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        ax.plot(self.y_true, label='True Labels', color='blue', linestyle='-.')
        ax.plot(self.y_pred, label='Predicted Labels', color='yellow')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Label (0: Normal, 1: Anomalous)')
        ax.set_title('True vs. Predicted Labels')
        ax.legend(loc=(1.05, 0.5))
        st.pyplot(fig)
        
        # confusion matrix
        st.divider()
        conf_matrix = metrics.confusion_matrix(self.y_true, self.y_pred)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax)
        plt.title('Confusion Matrix')
        st.pyplot(fig)
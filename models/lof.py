import streamlit as st

import matplotlib.pyplot as plt

from pyod.models.lof import LOF

from sklearn import metrics

class lof():
    def __init__(self, train_data, test_data, y_true) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.y_true = y_true

        print('training', len(self.train_data))
        print('testing', len(self.test_data))


        clf = LOF()
        clf.fit(self.train_data)
        self.y_pred = clf.fit_predict(self.test_data)

        for i in range(len(self.y_pred)):
            if self.y_pred[i] == 1:
                self.y_pred[i] = 0
            else:
                self.y_pred[i] = 1

    def visualize(self):
        precision_lof = metrics.precision_score(self.y_true, self.y_pred)
        recall_lof = metrics.recall_score(self.y_true, self.y_pred)
        f1_score_lof = metrics.f1_score(self.y_true, self.y_pred)
        fpr, tpr, _ = metrics.roc_curve(self.y_true, self.y_pred)
        auc_roc_lof = metrics.auc(fpr, tpr)

        print("Metrics: ")
        print('Precision: {:.4f}'.format(precision_lof))
        print('Recall: {:.4f}'.format(recall_lof))
        print('F1 score: {:.4f}'.format(f1_score_lof))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_lof))

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'AUC = {auc_roc_lof:.4f}')
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

from pyod.models.lof import LOF

from sklearn import metrics

class lof():
    def __init__(self, train_data, test_data, y_true) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.y_true = y_true

    def train_test(self):
        clf = LOF()
        clf.fit(self.train_data)
        y_pred = clf.fit_predict(self.test_data)

        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        precision_lof = metrics.precision_score(self.y_true, y_pred)
        recall_lof = metrics.recall_score(self.y_true, y_pred)
        f1_score_lof = metrics.f1_score(self.y_true, y_pred)
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_lof = metrics.auc(fpr, tpr)

        print("Metrics: ")
        print('Precision: {:.4f}'.format(precision_lof))
        print('Recall: {:.4f}'.format(recall_lof))
        print('F1 score: {:.4f}'.format(f1_score_lof))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_lof))

        return self.y_true, y_pred, fpr, tpr, auc_roc_lof

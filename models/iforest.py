from sklearn.ensemble import IsolationForest
from sklearn import metrics

class iForest():
    def __init__(self, train_data, test_data, y_true) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.y_true = y_true


    def train_test(self):
        clf = IsolationForest(n_estimators=100, max_samples='auto', 
                              contamination=float(0.12), bootstrap=False, 
                              n_jobs=-1, random_state=42, verbose=0)
        clf.fit(self.train_data)
        y_pred = clf.predict(self.test_data)

        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        precision_iforest = metrics.precision_score(self.y_true, y_pred)
        recall_iforest = metrics.recall_score(self.y_true, y_pred)
        f1_score_iforest = metrics.f1_score(self.y_true, y_pred)
        fpr, tpr, _ = metrics.roc_curve(self.y_true, y_pred)
        auc_roc_iforest = metrics.auc(fpr, tpr)

        print("Metrics: ")
        print('Precision: {:.4f}'.format(precision_iforest))
        print('Recall: {:.4f}'.format(recall_iforest))
        print('F1 score: {:.4f}'.format(f1_score_iforest))
        print('AUC-ROC socre: {:.4f}'.format(auc_roc_iforest))

        return self.y_true, y_pred, fpr, tpr, auc_roc_iforest

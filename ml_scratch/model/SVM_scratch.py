# SVCクラスをインポート
from sklearn.svm import SVC

# SVM_scratchクラスを定義
class SVM_scratch:
    
    # インスタンス作成時にSVCクラスのインスタンスを作成
    def __init__(self):
        self.cls = SVC()

    # SVCクラスのfitメソッドを使用し、
    # 引数のデータから学習モデルを作成しreturnする
    def fit(self, X_train, y_train):
        self.cls.fit(X_train, y_train)
        return self.cls

    # SVCクラスのpredictメソッドにて予測を行い結果をreturnする
    def predict(self, X_test):
        pre = self.cls.predict(X_test)
        return pre
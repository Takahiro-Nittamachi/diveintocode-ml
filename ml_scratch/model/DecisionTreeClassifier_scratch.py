# DecisionTreeClassifierクラスをインポート
from sklearn.tree import DecisionTreeClassifier

# DecisionTreeClassifier_scratchクラスを定義
class DecisionTreeClassifier_scratch:
    
    # インスタンス作成時にDecisionTreeClassifierクラスのインスタンスを作成
    def __init__(self):
        self.cls = DecisionTreeClassifier()

    # DecisionTreeClassifierクラスのfitメソッドを使用し、
    # 引数のデータから学習モデルを作成しreturnする
    def fit(self, X_train, y_train):
        self.cls.fit(X_train, y_train)
        return self.cls

    # DecisionTreeClassifierクラスのpredictメソッドにて予測を行い結果をreturnする
    def predict(self, X_test):
        pre = self.cls.predict(X_test)
        return pre
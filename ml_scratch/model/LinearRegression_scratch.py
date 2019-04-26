# LinearRegressionクラスをインポート
from sklearn.linear_model import LinearRegression

# LinearRegression_scratchクラスを定義
class LinearRegression_scratch:
    
    # インスタンス作成時にLinearRegressionクラスのインスタンスを作成
    def __init__(self):
        self.cls = LinearRegression()

    # LinearRegressionクラスのfitメソッドを使用し、
    # 引数のデータから学習モデルを作成しreturnする
    def fit(self, X_train, y_train):
        self.cls.fit(X_train, y_train)
        return self.cls

    # LinearRegressionクラスのpredictメソッドにて予測を行い結果をreturnする
    def predict(self, X_test):
        pre = self.cls.predict(X_test)
        return pre
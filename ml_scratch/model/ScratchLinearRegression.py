# 各クラスをインポート
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""
クラス定義
"""
class ScratchLinearRegression():

    """
    線形回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録
      """
    

    def __init__(self, num_iter, lr, bias=True, verbose=False):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.coef = 1
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        
    
    def linear_hypothesis(self,X):
    # def _linear_hypothesis(self,X):
        """
        def _linear_hypothesis(self, X):
        線形の仮定関数を計算する
        
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
        学習データ
        
        Returns
        -------
        次の形のndarray, shape (n_samples, 1)
        線形の仮定関数による推定結果
        """
        
        """
        バイアス項がある時の処理
        →入力されたnumpy配列Xの左端の列に
            shape(n_sample, 1)のnumpy配列を結合
        """
        
        # バイアス項がある時
        if self.bias == False:
            
            # バイアス項有り→左端の列にshape(n_sample, 1)のnumpy配列を挿入
            X = np.insert(X, 0, 1, axis=1)
        
        """
        仮定関数を生成
        """
        # 初期パラメータを設定(Xの行数の数だけ乱数配列を作成)
        np.random.seed(0)
        self.coef = np.random.rand(X.shape[1])
        
        # パラメータのshapeを(1,Xの行数)　に変形
        self.coef = self.coef.reshape(-1,1)
        
        # self.coef　と　X とのドット積を計算
        return np.dot(X, self.coef)
    
    
    def MSE(self, y_pred, y):
        """
        平均二乗誤差の計算

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
          推定した値
        y : 次の形のndarray, shape (n_samples,)
          正解値

        Returns
        ----------
        mse : numpy.float
          平均二乗誤差
        """
        
        # y_predの行数
        number_of_data = len(y_pred)
        
        # 平均二乗誤差の計算
        mse = np.sum((y_pred - y) ** 2) / (number_of_data * 2)
        
        return mse

    
    def gradient_descent(self, X, error):
        """
        def gradient_descent(self, X, error):
        最小降下法により最適なパラメータを計算する
        
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
        学習データ
        
        error : 誤差(float)
        学習データ
        
        Returns
        -------
        なし
        """
        a = np.sum(error * X)
        b = self.lr / len(X)
        
        # パラメータを更新
        self.coef = self.coef - ((np.sum(error * X)) * (self.lr / len(X)))
    
        return


    def fit(self, X, y, X_val=None, y_val=None):
        
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        # 変数を初期化
        y_pred = 0
        error = 0
        self.coef = 0
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        
        # 入力Xとyをnumpy配列に変換(dataframeを入力した際の対処)
        X = np.array(X)
        y = np.array(y)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        """
        入力値が１次元の時のshape数の変更作業
        """
        # 入力　X が１次元か？
        if X.ndim == 1:
            
            # 1次元の場合→shapeを(1, n_sample)に変更
            X = X[:, np.newaxis]
        
        # 入力　y が１次元か？
        if y.ndim == 1:
            
            # 1次元の場合→shapeを(1, n_sample)に変更
            y=y[:,np.newaxis]
        
        # 入力　X_val が１次元か？
        if X_val.ndim == 1:
            
            # 1次元の場合→shapeを(1, n_sample)に変更
            X_val = X_val[:, np.newaxis]
        
        # 入力　y_val が１次元か？
        if y_val.ndim == 1:
            
            # 1次元の場合→shapeを(1, n_sample)に変更
            y_val = y_val[:, np.newaxis]
            
        """
        仮定関数の計算
        """
        # 仮定関数を計算
        y_pred = self.linear_hypothesis(X)
        
        """
        パラメータ更新
        """
        #   num_iter 回繰り返し
        for i in range(self.iter):

            # 予測値と正解値の誤差を計算
            error = y_pred - y

            # 検証用の正解データがあるなら
            if y_val is not None:
                # 予測値との誤差を計算し、numpy配列に格納
                self.val_loss[i] = self.MSE(y_pred, y_val)
                
            
            # 予測値と学習用の正解データとの誤差を計算しnumpy配列に格納
            self.loss[i] = self.MSE(y_pred, y)

            # パラメータを更新 
            self.gradient_descent(y_pred, error)
            
            # 仮定関数を更新
            y_pred = np.dot(X, self.coef) 

            #verboseをTrueにした際は学習過程を出力
            if self.verbose == True:
                print('Epoch数：{}　MSE(学習データ)：{} MSE(検証データ):{}\n'.format(i, self.loss[i], self.val_loss[i]))
            
        return
    

    def predict(self, X):
        
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """
        
        if X.ndim == 1:
            # 1次元の場合→shapeを(1, n_sample)に変更
            X_val = X_val[:, np.newaxis]
        return np.dot(X, self.coef)
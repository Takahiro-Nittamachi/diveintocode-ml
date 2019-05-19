# 各クラスをインポート
import numpy as np
import pandas as pd

"""
クラス定義
"""
class ScratchLogisticRegression():

    """
    線形回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    lam : float
      正則化パラメータ
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
    

    def __init__(self, num_iter, lr, lam, bias=True, verbose=False):
        """
        ロジスティック回帰のスクラッチ実装

        Parameters
        ----------
        num_iter : int
          イテレーション数
        lr : float
          学習率
        lam : float
            正則化項の効果の効き目を設定するパラメータ
        no_bias : bool
          バイアス項を入れない場合はTrue
        verbose : bool
          学習過程を出力する場合はTrue

        Attributes
        ----------
        self.loss : 次の形のndarray, shape (self.iter,)
          学習用データに対する損失の記録
        self.val_loss : 次の形のndarray, shape (self.iter,)
          検証用データに対する損失の記録
        self.iter_ : 次の形のndarray, shape (n_features,)
          パラメータ
        self.y_pred_ : 次の形のndarray, shape (n_features, )
          パラメータ
        self.coef_ : 次の形のndarray, shape (1, n_features)
          パラメータ
        self.n_sample_ : int
          データのサンプル数
        self.n_feature_ : int
          特徴量の数
        self.ver_exist_ : Bool
          検証用データの有無
        """
        
        
        self.iter = num_iter
        self.lr = lr
        self.lam = lam
        self.bias = bias
        self.loss = None # 目的関数(対学習データの正解値)の計算結果を格納するnumpy配列
        self.val_loss = None # 目的関数(対検証データの正解値)の計算結果を格納するnumpy目的関数
        self.y_pred_ = None
        self.coef_ = None
        self.n_sample_ = None
        self.n_feature_ = None
        self.ver_exist_ = None # 検証用データの有無

        return
    
    
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        学習する

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
        
        
        
        
        
        self.coef_ = 0
        self.n_sample_ = 0
        self.n_feature_ = 0
        self.ver_exist_ = None
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        
        # データの前処理
        #  入力データのnumpy配列化、検証用データの有無の確認、
        # バイアス項の有無を確認し、有りの場合はXに１列追加、入力Xの行数・列数取得
        X, y, X_val, y_val = self.preprocessing(X, y, X_val, y_val)
        
        
        # 暫定の予測値の計算
        self.y_pred_ = self.linear_hypothesis(X)
              
        # 指定回数繰り返し
        for i in range(self.iter):
            self.loss[i] = self.cost(y)
            
            if self.ver_exist_:
                self.val_loss[i] = self.cost(y_val)
            
            self.gradient_descent(X, y)
            
            self.y_pred_ = self.sigmoid(np.dot(X, self.coef_))
  

    
        return
   

    def predict(self, X):
        """
        分類結果を

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
            
        Return
        ----------
        分類結果
        np.where(self.sigmoid(np.dot(X, self.coef_)) > 0.5 ,1,-1)
        """
        
        
        # 入力Xをnumpy配列に変換
        X = np.array(X)
        
        # 入力Xが１次元なら２次元に変換
        if X.ndim == 1:
            X = X.reshape(1,-1)
            
        # バイアス項の有無を確認
        if self.bias == False:
            # バイアス項有り→左端の列に値が1　shape(n_sample, 1)のnumpy配列を追加
            X = np.insert(X, 0, 1, axis=1)
        
        # 分類結果を返す
        return np.where(self.sigmoid(np.dot(X, self.coef_)) > 0.5 ,1,-1)
    
    
    def preprocessing(self, X, y, X_val, y_val):
        """
        データの前処理を行う

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
            
        Return
        ----------
        """

        """
        入力データをnumpy配列化、１次元データは２次元化
        """
        # 入力Xとyをnumpy配列に変換(dataframeを入力した際の対処)
        X = np.array(X)
        y = np.array(y)
        
        # 検証用データが１次元データか判別
        # １次元の場合→２次元データに変換　shapeを(1, n_sample)に変更
        if X.ndim == 1:
            X = X[:, np.newaxis]
        
        if y.ndim == 1:
            y=y[:,np.newaxis]
            
            
        """
        検証用データの有無を確認し、numpy配列化、１次元データは２次元化
        """


        # 検証用データの有無のフラグ
        if X_val is not None and y_val is not None:
            
            # 検証用データの有無のフラグを設定
            self.ver_exist_ = True
            
            # 検証用のデータがある　→　numpy配列に変換
            X_val = np.array(X_val)
            y_val = np.array(y_val)

            # 検証用データが１次元データか判別
            # １次元の場合→２次元データに変換　shapeを(1, n_sample)に変更
            if X_val.ndim == 1:
                X_val = X_val[:, np.newaxis]

            if y_val.ndim == 1:
                y_val = y_val[:, np.newaxis]
                
        
        """
        バイアス項の有無を確認し、有の場合は学習用の特徴量データXの左端にバイアス項用の列を追加
        """
        # バイアス項の有無を確認
        if self.bias == False:
            # バイアス項有り→左端の列に値が1　shape(n_sample, 1)のnumpy配列を追加
            X = np.insert(X, 0, 1, axis=1)

            
            
        # 入力データのサンプル数、特徴量数を取得
        self.n_sample_ = len(X)
        self.n_feature_ = X.shape[1]
        
        return X, y, X_val, y_val
 

    def linear_hypothesis(self, X):
        """
        初期の予測値を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
            
        Return
        ----------
        """
        # 初期パラメータを設定(Xの行数の数だけ乱数配列を作成)
        np.random.seed(0)
        self.coef_ = np.random.rand(X.shape[1])
        
        # 2次元配列に変更
        self.coef_ = self.coef_[:, np.newaxis]
        
        # self.coef　と　X とのドット積をシグモイド関数に代入した値を計算
        return self.sigmoid(np.dot(X, self.coef_))
    
    
    def sigmoid(self, X):
        """
        シグモイド関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
            
        Return
        ----------
        シグモイド関数の計算結果
        """
        
        return 1/(1 + np.exp(- X))


    
    def cost(self, y):
        """
        目的関数を計算する

        Parameters
        ----------
        y : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
            
        Return
        ----------
        シグモイド関数の計算結果
        """
        cost = (np.sum(- y * np.log(self.y_pred_) - (1 - y) * np.log(1 - self.y_pred_)) / self.n_sample_) + \
                    (self.lam * np.sum(self.coef_ ** 2) / (2 * self.n_sample_))
        return cost
    

    def gradient_descent(self, X, y):
        """
        パラメータを更新する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
            
        Return
        ----------
        シグモイド関数の計算結果
        """
        
        # 勾配の計算
        theta = np.dot(X.T, self.y_pred_ - y) / self.n_sample_

        # 正則化項と内積をとる値1　shape(self.n_feature, 1)のnumpy配列
        flag = np.ones(self.n_feature_).reshape(-1,1)
        
        # バイアス項の有無を確認
        if self.bias == False:
            # バイアス項有り→一番上の値[0,0]を0とする。それ以外は1のまま。
            flag[0,0] = 0

        
        # 正則化項の計算
        #　バイアス項、無しの場合　→　flagの中身は全て1の為、値は変化せず
        # バイアス項、有りの場合　→　[0,0]の値(シータゼロの正則化項と内積をとる値)のみ0の為、
        # シータゼロのみ正則化項の部分がゼロとなり、
        reg_term = (self.lam * self.coef_ / self.n_sample_) * flag

        
        # パラメータの更新
        self.coef_ = self.coef_ - self.lr * (theta + reg_term)

        return
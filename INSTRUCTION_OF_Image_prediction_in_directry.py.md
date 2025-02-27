# はじめに
ここではsam2によるマスク画像の生成を行う際に，対象物のアノテーションデータを利用することで，精度を上げる（マスクが分割されていたり，欠けたりしていない）方法について説明します．

# 環境構築
sam2の導入は公式の[README.md](https://github.com/facebookresearch/sam2?tab=readme-ov-file)か[INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)に従って行います．

# 静止画像にマスクを生成する方法
基本的には[このドキュメント](https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)に準拠して説明します．
sam2にはマスクを生成する対象物を選択る方法が以下の種類あります．

## 1. はじめに
ここではSAM2によるマスク画像の生成を行う際に，対象物のアノテーションデータを利用することで，精度を上げる（マスクが分割されていたり，欠けたりしていない）方法について説明します．
人間の全身を対象としたセグメンテーションでは，**手や足など一部が欠ける問題**が発生しやすく，そうした問題を防ぐための**アノテーションデータ（バウンディングボックス＋キーポイント）を活用したアプローチ**について，[Image_prediction_in_directly.py](https://github.com/Tomoki-Ida/sam2/blob/266920633da59471aa3d0f36f8ae41555e8f480f/Image_prediction_in_directly.py)をもとに解説します．


## 2. 環境構築
SAM2の導入は公式の[README.md](https://github.com/facebookresearch/sam2?tab=readme-ov-file)および[INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)に従って行います．

以下のライブラリが必要です．

- Python 3.8 以上
- PyTorch 1.13 以上
- torchvision
- matplotlib
- numpy
- opencv-python
- PIL (Pillow)

CUDAデバイス（GPU）の使用を推奨します．


## 3. 静止画像にマスクを生成する方法

基本的には[このドキュメント](https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)に準拠します．

モジュールの導入および関数の定義をドキュメントに従い行ってください．

SAM2にはマスクを生成する際の対象物の選択方法として，以下の3つの方法があります．

- **ポイント指定**：対象物の一部に点を打ち，そこを起点にマスクを生成
- **バウンディングボックス指定**：対象物全体を囲う枠を与えて，内部をマスクとして生成
- **キーポイント指定**：対象物の複数部位（手・足・頭など）に点を打ち，それらを全て含むマスクを生成

以下，それぞれの方法について解説します．


### 3.1 ポイント指定によるマスク生成

特定の位置に点（座標）を指定し，その周囲のオブジェクトを識別してマスクを生成する方法です．

このスクリプトでは，画像の中央に点を打ち，その位置を基準にマスクを作成しています．

```python
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 予測器を設定
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)

# オブジェクト上の座標を指定 (例: 画像の中心)
input_point = np.array([[image.shape[1] * 0.5, image.shape[0] * 0.5]])
input_label = np.array([1])  # 1 = 前景点

# マスク予測
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False
)

masks.shape  # (number_of_masks) x H x W

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)
```

### 3.2 バウンディングボックス指定によるマスク生成
対象物の領域全体を指定することで，特定のオブジェクトを囲みながらマスクを生成する方法です．
この方法では，対象物の範囲を手動で設定することで，正確なマスクを作成できます．

```python
# オブジェクトのバウンディングボックスを指定 (x1, y1, x2, y2)
input_box = np.array([50, 30, 400, 350])  # 例として左上(50,30)から右下(400,350)までの範囲

# マスク予測
masks, scores, _ = predictor.predict(
    box=input_box[None, :],
    multimask_output=False
)
```


### 3.3 キーポイント指定によるマスク生成
特定のオブジェクトの複数の部位を指定することで，対象物全体をより詳細に認識してマスクを生成する方法です．
キーポイントを複数指定することで，特に手足や小さなオブジェクトが欠ける問題を軽減できます．

```python
# キーポイントを指定 (複数点)
input_points = np.array([
    [100, 150],  # 例: 頭部
    [120, 300],  # 例: 胴体
    [90, 450],   # 例: 足
])
input_labels = np.array([1, 1, 1])  # すべて前景点として指定

# マスク予測
masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False
)
```


### 3.4 背景点による非対象物の除去
対象物だけを抽出したい場合，背景にラベルを0とした「背景点（background point）」を打つことで，**対象外部分を明示的に除外**する機能も存在します．
これにより，不要なオブジェクトを除外しつつ，ターゲットであるオブジェクトのマスク精度を向上させることができます．
ただし，今回のスクリプト「Image_prediction_in_directly.py」では，この機能は使用していません．

```python
# 前景点と背景点の指定
input_point = np.array([[500, 375], [1125, 625]])  # 例: 500,375 は対象物，1125,625 は背景
input_label = np.array([1, 0])  # 1 = 前景点，0 = 背景点

# マスク予測（背景点を考慮）
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False
)
```


## 4. 追加した機能について（Image_prediction_in_directly.py）

### 4.1 画像ごとのアノテーションデータの読み込み

COCO形式のアノテーションJSONから，画像ごとのバウンディングボックスとキーポイントを読み込みます．  
画像名とアノテーション情報を対応付ける部分は，以下のように実装しています．

```python
# JSONファイルからバウンディングボックスとキーポイントを読み込む
def load_annotations(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # image_id から file_name へのマッピングを作成
    id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    annotations = {}
    for entry in data['annotations']:
        image_id = entry['image_id']
        file_name = id_to_filename.get(image_id)
        if file_name:
            bbox = entry['bbox']   # 'bbox' は [x1, y1, x2, y2] の形式であると仮定
            keypoints = entry.get('keypoints', [])  # 'keypoints' が JSON に含まれていると仮定
            annotations[file_name] = {'bbox': bbox, 'keypoints': keypoints}
    return annotations
```


### 4.2 キーポイントの可視性処理

キーポイントは，可視性フラグ（visibility）によって利用可否を判定しています．  
以下の関数で，可視な座標のみ抽出しています．

```python
# キーポイントを処理し，座標と可視性を抽出する
def process_keypoints(keypoints):
    coords = []
    labels = []
    for keypoint in keypoints:
        if isinstance(keypoint, list) and len(keypoint) == 3:
            x, y, v = keypoint
            if v > 0:  # 可視な座標のみを使用
                coords.append((x, y))
                labels.append(1)  # 前景点としてラベル付け
    return coords
```


### 4.3 画像へのアノテーション描画（BBox・Keypoints）

推論結果を保存する際に，確認のためにバウンディングボックスとキーポイントを描画しています．

```python
# バウンディングボックスとキーポイントを重畳した画像を保存
def save_bbox_and_keypoints_overlay(image, bbox, keypoints, output_path):
    # コピーした画像にバウンディングボックスを描画
    image_with_annotations = image.copy()
    x1, y1, x2, y2 = map(int, bbox)

    # 可視のキーポイントを描画
    cv2.rectangle(image_with_annotations, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x, y in keypoints:
        cv2.circle(image_with_annotations, (int(x), int(y)), 5, (0, 0, 255), -1)

    # 画像を保存
    cv2.imwrite(output_path, image_with_annotations)
```


### 4.4 マスク画像へのオーバーレイ保存

生成したマスクを元画像にオーバーレイして保存します．

```python
def show_image_with_mask(image, mask, output_path):
    # 図（画像）の準備
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 表示のためにRRGBに変換
    show_mask(mask, ax)
    ax.axis('off')
    plt.tight_layout()

    # 可視化画像を保存
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
```


### 4.5 推論処理と時間計測

画像ごとの推論時間を計測し，最後に平均時間を表示する仕組みです．

```python
start_time = time.time()

# 推論処理
predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=np.array(keypoints),
    point_labels=np.ones(len(keypoints)),
    box=input_box[None, :],
    multimask_output=False,
)

end_time = time.time()
prediction_time = end_time - start_time
prediction_times.append(prediction_time)
```


### 4.6 まとめ

「Image_prediction_in_directly.py」は，事前アノテーション（バウンディングボックス＋キーポイント）を活用して，高精度なセグメンテーションを実現するスクリプトです．
人間を対象としたときに手足などが欠ける問題を防ぐために書いたものですが，物体セグメンテーションにも応用可能です．

## 5. ディレクトリ構成

```
project/
├── sam2/
│   ├── build_sam.py
│   ├── sam2_image_predictor.py
│   ├── ...
├── checkpoints/
│   └── sam2.1_hiera_large.pt
├── configs/
│   └── sam2.1/
│       └── sam2.1_hiera_l.yaml
├── Image_prediction_in_directly.py
├── data/
│   ├── images/
│   ├── annotations/
│       └── keypoints.json
├── results/
│   ├── mask_overlay/
│   ├── bbox_overlay/
```

## 6. 実行例

```bash
python Image_prediction_in_directly.py
```

画像フォルダやJSONのパスはスクリプト内で指定しています．必要に応じて変更してください．

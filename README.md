## Dynamic Split Computingのための混合精度ニューラルネットワーク最適化手法


### インストール

1. フォルダ内のDockerファイルより環境をインストールし、ImageNet datasetをダウンロードする[official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet)。 ImageNet-100 datasetを作成する。（https://github.com/danielchyeh/ImageNet-100-Pytorch）

2. リポジトリをクローンする。



### DSC-aware mixed-precision NASによる混合精度ネットワークの探索

DSC-aware mixed-precision NASのトレーニングを始める。（例：EfficientNetV1-B0）
```
python search.py \
  -a mixeffnet_b0_w2468a2468_100 --epochs 50 --step-epoch 20 --lr 0.1 --lra 0.01 --cd 0.0001 --csd 0.01 \
  -j 16 -b 80 --save-dir mixeffnet_b0_w2468a2468_100_csd0.01 /datasets/imagenet-100
```

その他のアーキテクチャとしては、EfficientNetV1-B3がある。



### 探索された混合精度ネットワークの訓練

DSC-aware mixed-precision NASの検索が終了したら、学習したビット割り当てで分類モデルの学習を開始する。 
```
python main.py \
  -a quanteffnet_cfg_2468 --epochs 100 --step-epoch 30\
  --ac --ac /EdMIPS/arch_output/mixeffnet_b0_w2468a2468_100_csd0.01/arch_model_best.pth.tar \ 
  -j 16 -b 120 --gpu 0　--save-dir efficient_w2468a2468_100_csd0.01　/datasets/imagenet-100
```




### 注意事項

1. --cdは、computational complesity riskを表し、--csdは、split complesity riskを表している。これらを調整して最適なビット幅割り当てを実現する。



### DSC実験の実行（dynamic_split_computing.ipynb）

1. SELECTED_GPUS`（ノートブックの最初のセル）を修正して、使用する GPU を選択する。(※モバイルデバイス側のGPUはアンダークロックを行う)
2. それに応じて`config`（ノートブックの最後のセル）を修正し、実行する。
3. 結果のプロットを `inference_plot` ディレクトリに表示する。


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mixefficientnet import BasicCNNBlock




def get_natural_bottlenecks(model, input_size, compressive_only=True):  #EfficientNetのみ適応
    # 各層のinputサイズを計算して、圧縮率が最も高い層を探す
    natural_bottlenecks = []
    best_compression = 1.0
    cnn_count = 0  # CNNレイヤーのカウント
    input_bit = 8 # 入力のbit数
    min_bit = 2  # 探索する最小のbit数
    bit_compression = min_bit / input_bit

    device = next(model.parameters()).device
    
    mock_input = torch.randn(1, 3, input_size, input_size).to(device)
    previous_size = torch.prod(torch.tensor(mock_input.shape[1:])).item()

    for i, module in enumerate(model.features):
        # print(i, module)
        block_number = i-1 # 0はfeaturesの最初のBasicCNNBlockなので、1から始める
        if isinstance(module, BasicCNNBlock):
            print(f"Encountered BasicBlock at features.{i}")
            output = module(mock_input)
            mock_input = output.detach()
            continue
        
        input_size_layer = torch.prod(torch.tensor(mock_input.shape[1:])).item()
        if input_size_layer * min_bit < input_size * input_size * 3 * input_bit:
            compression = float(input_size_layer) / (input_size * input_size * 3)
            compression *= bit_compression
            if not compressive_only or compression < best_compression:
                natural_bottlenecks.append({
                    'layer_name': "block_{}".format(block_number),
                    'compression': compression,
                    'cnn_layer_number': cnn_count  # ここでCNNレイヤーの番号を記録
                })                                  #どうしてこっちでは'block_number'を記録しない？
                best_compression = compression
        output = module(mock_input)
        mock_input = output.detach()
        
        cnn_count += count_conv2d_layers(module)


    return natural_bottlenecks

def count_conv2d_layers(model):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count += 1
        elif isinstance(module, nn.Sequential):
            # Sequentialブロック内でさらにConv2dを探す
            for sub_module in module:
                if isinstance(sub_module, nn.Conv2d):
                    count += 1
    return count
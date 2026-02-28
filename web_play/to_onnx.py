import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NeuralNet import ResNet

def pt_to_onnx(game, model_path, onnx_output_path, num_resBlocks=4, num_hidden=64, input_channels=6):
    # 1. 모델 초기화 (학습 시 사용한 하이퍼파라미터로 수정하세요)
    # ResNet(game, num_resBlocks, num_hidden, device, input_channels=3)
    model = ResNet(game, num_resBlocks, num_hidden, 'cpu', input_channels=input_channels)
    
    # 2. 가중치 로드
    checkpoint = torch.load(model_path, map_location='cpu')
    # state_dict만 저장된 경우와 전체가 저장된 경우 대응
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    # 3. 더미 데이터 생성 (Batch, Channel, Height, Width)
    dummy_input = torch.randn(1, input_channels, game.column_count, game.row_count)

    # 4. ONNX 내보내기
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=20,
        input_names=['input'],
        output_names=['policy', 'value'],
        dynamic_axes={'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
    )
from Args import get_args
import argparse
import json
import torch

if __name__ == "__main__":
    """
    args = get_args("exp1")

    # save args as json for c++
    with open("config.json", "w") as f:
        json.dump(vars(args), f, indent = 4)
    """

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(6, 16, 3, padding=1)
            self.fc_p = torch.nn.Linear(16 * 7 * 7, 50) # Quoridor 액션 수에 맞춰 조정
            self.fc_v = torch.nn.Linear(16 * 7 * 7, 1)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            policy = torch.softmax(self.fc_p(x), dim=1)
            value = torch.tanh(self.fc_v(x))
            return policy, value

    model = torch.jit.script(DummyModel())
    model.save("test_model.pt")
    print("test_model.pt 생성 완료!")
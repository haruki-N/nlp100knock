import time
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_FEATURES=300   # 入力
HIDDEN_LAYER1 = 200 # 隠れ層1
HIDDEN_LAYER2 = 100 # 隠れ層2
OUTPUT_FEATURES=4   # 出力

class MLNet(nn.Module):
    def __init__(self):
        super(MLNet, self).__init__()
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN_LAYER1, bias=True)
        self.fc2 = nn.Linear(HIDDEN_LAYER1, HIDDEN_LAYER2, bias=True)
        self.fc3 = nn.Linear(HIDDEN_LAYER2, OUTPUT_FEATURES, bias=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    




def batch_trainee_gpu(batch_size, epoch_size=100):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    simple_net = Net().to(device)

    # 最適化手法の指定
    creterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(simple_net.parameters(), lr=0.01)

    # 学習の準備
    # trainのデータをdeviceに渡す
    x_train = torch.from_numpy(xi_vec_train).to(device)
    y_train = torch.from_numpy(yi_vec_train).to(device)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)

    # バッチサイズの指定
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 学習

    start_time = time.time()

    for epoch in range(epoch_size):  # エポック数分ループを回す

        for input_x, label_y in data_loader:
            # パラメータの勾配をリセット
            optimizer.zero_grad()

            # 順伝搬
            output = simple_net(input_x)
            loss = criterion(output, label_y)

            # 逆伝搬
            loss.backward()

            # パラメータ更新
            optimizer.step()

    time.sleep(0.01)
    run_time = time.time() - start_time
    print(f'Use device: {device}')
    print(f'Run Time  : {run_time}')
    print(f'Train: {get_accuracy(simple_net, xi_vec_train, yi_vec_train)}')
    print(f'Valid: {get_accuracy(simple_net, xi_vec_valid, yi_vec_valid)}')
    print(f'Test : {get_accuracy(simple_net, xi_vec_test, yi_vec_test)}')
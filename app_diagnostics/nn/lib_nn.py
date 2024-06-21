import torch
import torch.nn as nn
import config.lib_config as lib_config

class NeuralNetworkLSTM(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers, outputSize):
        super(NeuralNetworkLSTM, self).__init__()
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)
        self.seq = nn.Sequential(
            nn.Linear(hiddenSize, outputSize * 8),
            nn.ReLU(),  # для CE Loss
            nn.Linear(outputSize * 8, outputSize),
        )

    def forward(self, x):
        h0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        c0 = torch.zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.seq(out)
        return out

    def train_model(self, trainLoader, testLoader=None, numEpochs=100):
        criterion = nn.CrossEntropyLoss()
        #lr = 0.001
        optimizer = torch.optim.Adam(self.parameters())

        self.train()  # переводим модель в режим обучения
        for epoch in range(numEpochs):
            for inputs, targets in trainLoader:
                optimizer.zero_grad()
                # print(inputs.shape)
                # exit()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{numEpochs}], Loss: {loss.item():.4f}')

        # Оцениваем нейросеть на тестовых данных
        with torch.no_grad():  # отключаем вычисление градиентов
            t_0, t_1, t_2, f_0, f_1, f_2 = 0, 0, 0, 0, 0, 0
            for inputs, targets in testLoader:
                outputs = self(inputs)
                _, result = torch.max(outputs, dim=1)
                for i in range(0, outputs.shape[0]):
                    if targets[i][result[i]] == 1:
                        if (result[i] == torch.tensor(0)):
                            t_0 += 1
                        if (result[i] == torch.tensor(1)):
                            t_1 += 1
                        if (result[i] == torch.tensor(2)):
                            t_2 += 1
                    else:
                        if (targets[i][0] == torch.tensor(1)):
                            f_0 += 1
                        if (targets[i][1] == torch.tensor(1)):
                            f_1 += 1
                        if (targets[i][2] == torch.tensor(1)):
                            f_2 += 1
            print(f't_0: {t_0},t_1: {t_1},t_2: {t_2}, t_all: {t_0+t_1+t_2},f_0: {f_0}, f_1: {f_1}, f_2: {f_2}, f_all: {f_0+f_1+f_2}')
            accuracy = (t_0+t_1+t_2)/(t_0+t_1+t_2+f_0+f_1+f_2)
            print("accuracy = ", accuracy)

    def predict_model(self, inputData):
        self.eval()  # переводим модель в режим предсказания
        with torch.no_grad():
            dataToNN = torch.tensor(inputData, dtype=torch.float32)
            outputs = self(dataToNN)
            _, result = torch.max(outputs, dim=1)
        return result

    def save_model(self, file_path='nn_weights.pth'):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path='nn_weights.pth'):
        self.load_state_dict(torch.load(file_path))

# Функция для запуска нейросети и тренировки, если модель изменена
def start_nn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nn = NeuralNetworkLSTM(inputSize=4, hiddenSize=64, numLayers=1, outputSize=1).to(device)
    configFromFile = lib_config.ConfigJSON(lib_config.CONFIG_PATH)
    allDicts = configFromFile.get_dicts()
    nnDict = allDicts['nnData']
    nn.load_state_dict(torch.load(nnDict['nnName']))

#train_nn()
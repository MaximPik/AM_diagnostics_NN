import numpy as np
import matplotlib.pyplot as plt
import threading
import tkinter as tk
import opclabs_quickopc
from OpcLabs.EasyOpc.DataAccess import *
from OpcLabs.EasyOpc.OperationModel import *

class OPCServer():
    def __init__(self, classNN,arrMinVal, arrMaxVal, axs, canvas, listObj, OPCDict):
        self.client = EasyDAClient()
        self.timeout = 10  # таймаут в секундах
        self.allData = None
        self.listObj = listObj
        self.listObj.insert(tk.END, f"Начало соединения с OPC")
        clientThread = threading.Thread(target=self.handle_client, args=(classNN, arrMinVal, arrMaxVal, axs, canvas,))
        clientThread.start()
        # Запуск таймера
        self.timer = threading.Timer(self.timeout, self.server_close)
        self.timer.start()
        self.speed = OPCDict['speed']
        self.current = OPCDict['current']
        self.freq = OPCDict['freq']
        self.opcName = OPCDict['opcName']

    def handle_client(self, classNN, arrMinVal, arrMaxVal, axs, canvas):
        for ax in axs:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            ax.clear()  # Очистка каждого подграфика при новом подключении
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
        dataArray = []
        self.allData = []
        while True:
            speed = IEasyDAClientExtension.ReadItemValue(self.client, '', self.opcName, self.speed)
            current = IEasyDAClientExtension.ReadItemValue(self.client, '', self.opcName, self.current)
            out_freq = IEasyDAClientExtension.ReadItemValue(self.client, '', self.opcName, self.freq)
            if current and speed:
                values = (speed, out_freq/10, current/10)
                dataArray.append(values)
                if len(dataArray) == 4:
                    XDataArray = np.array(dataArray)
                    # Составляем массив минимальных и максимальных значений
                    maxValues = np.array(arrMaxVal)
                    minValues = np.array(arrMinVal)
                    # Реализуем StandardScaler [-1;1]
                    for j in range(XDataArray.shape[1]):  # проходимся по каждому столбцу
                        minVal = minValues[j]
                        maxVal = maxValues[j]
                        # применяем формулу нормирования от -1 до 1
                        XDataArray[:, j] = (XDataArray[:, j] - minVal) / (maxVal - minVal) * 2 - 1
                    dataArray = dataArray[1:]
                    prediction = classNN.predict_model([XDataArray])
                    self.allData.append(values + (prediction.item(),))
                    print(f"allData: {self.allData[-1]}")
                else:
                    self.allData.append(values + (0,))
                for iter in range(len(self.allData[0])):
                    axs[iter].plot(np.arange(len(self.allData)), [val[iter] for val in self.allData], color='r')
                canvas.draw()
                plt.pause(0.01)
                # Перезапуск таймера
                if self.timer:
                    self.timer.cancel()
                self.timer = threading.Timer(self.timeout, self.server_close)
                self.timer.start()
    
    def server_close(self):
        #print("Connection timed out, closing server")
        self.listObj.insert(tk.END, f"Connection timed out, closing server")
        # self.server_socket.close()
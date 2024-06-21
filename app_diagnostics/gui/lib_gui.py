import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import common.lib_common as lib_common
import config.lib_config as lib_config
import server.lib_server as lib_server

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        #Добавим заголовок окна
        self.title("Engine diagnostics")
        #Создаём вкладку
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")
        engineDiagnostics = Interface(self.notebook, tabTitle="Graphics")
        self.notebook.add(engineDiagnostics, text=engineDiagnostics.tabTitle)
        self.geometry("1200x700+{}+{}".format(self.winfo_screenwidth() // 2-600, self.winfo_screenheight() // 2 -350))
        self.protocol("WM_DELETE_WINDOW") #Закрытие программы

class Interface(ttk.Frame):
    def __init__(self, parent, tabTitle):
        super().__init__(parent)
        super().__init__()
        self.tabTitle = tabTitle
        #####Ниже идёт описание интерфейса##########
        #Добавляем кнопку "Старт"
        startButton = ttk.Button(self, text="Start", command=self.start_pressed)
        startButton.config(width=25)
        startButton.place(relx=0.03,rely=0.05, anchor="nw")
        #Добавляем кнопку "Стоп"
        stopButton = ttk.Button(self, text="Stop", command=self.stop_pressed)
        stopButton.config(width=25)
        stopButton.place(relx=0.03,rely=0.1, anchor="nw")
        #Добавляем поле, где будет писаться состояние программы
        self.stateList = tk.Listbox(self)
        self.stateList.place(relx=0.01, rely=0.2,anchor="nw")
        self.stateList.configure(width=30, state='normal')
        # Добавляем поле для графика в интерфейс
        # Создаем фигуру и оси для графиков
        fig = Figure(figsize=(8, 6), dpi=100)
        self.axs = []
        allDicts = lib_config.setup_config()
        for iter in range(1, 5):
            self.axs.append(fig.add_subplot(320+iter))
            self.axs[-1].set_ylabel(allDicts["fileData"]["allAxisXName"][iter - 1])
            self.axs[-1].set_xlabel("number of points")
            #Выравниваем графики
            box = self.axs[-1].get_position()
            self.axs[-1].set_position([box.x0, box.y0, box.width, box.height * 0.8])

        # Создаем холст для рисования графика
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        # Добавляем холст в окно
        self.canvas.get_tk_widget().place(relx=0.17, rely=0.05, anchor="nw",relwidth=0.9)

    def start_pressed(self):
        nn, scalarDict, OPCDict = lib_common.setup() #From common lib
        self.serv = lib_server.OPCServer(nn, scalarDict['minValues'], scalarDict['maxValues'],
                                 self.axs, self.canvas, self.stateList, OPCDict)

    def stop_pressed(self):
        #self.serv.server_close()
        pass
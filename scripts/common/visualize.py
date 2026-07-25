import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from enum import Enum
from dataclasses import dataclass

class Visualize():
    class Action(Enum):
        TWOVAR_CMAP = 1
        ONEVAR_PLOT = 2
        ONEVAR_SCATTER = 3
        
    class Graph():
        def __init__(self, action, input, output, overwrap=False, title="", datalabel="", xlabel="x", ylabel="y", color=""):
            self.action = action
            self.input = input
            self.output = output
            self.overwrap = overwrap
            self.title = title
            self.datalabel = datalabel
            self.xlabel = xlabel
            self.ylabel = ylabel
            self.color = color
    
    def __init__(self):
        self.fig = None
        self.graphs = []
        self.figsize=(8, 6)
        self.recording = False
        self.ani_filename = None
        self.ani_interval = 200
        self.ims = []
        
    def draw_2_variable_func_colormap(self, f, x_range, y_range, title="", xlabel="x", ylabel="y", n=200):
        x = np.linspace(x_range[0], x_range[1], n)
        y = np.linspace(y_range[0], y_range[1], n)
        X, Y = np.meshgrid(x, y)
        Z = [[f(x__, y__) for x__, y__ in zip(x_, y_)] for x_, y_ in zip(X, Y)]
        self.graphs.append(self.Graph(self.Action.TWOVAR_CMAP, [X, Y], Z, False, title, "", xlabel, ylabel))
        
    def draw_2_variable_data_colormap(self, x, y, z, title="", xlabel="x", ylabel="y"):
        self.graphs.append(self.Graph(self.Action.TWOVAR_CMAP, [x, y], z, False, title, "", xlabel, ylabel))

    def draw_1_variable_func(self, f, x_range, overwrap=False, title="", datalabel="", xlabel="x", ylabel="y", color="", n=200):
        x = np.linspace(x_range[0], x_range[1], n)
        y = [f(x_) for x_ in x]
        self.graphs.append(self.Graph(self.Action.ONEVAR_PLOT, x, y, overwrap, title, datalabel, xlabel, ylabel, color))
        
    def draw_1_variable_data(self, x, y, overwrap=False, title="", datalabel="", xlabel="x", ylabel="y", color=""):
        self.graphs.append(self.Graph(self.Action.ONEVAR_SCATTER, x, y, overwrap, title, datalabel, xlabel, ylabel, color))

    def reset(self):
        self.__init__()

    def num_of_sheets(self, graphs):
        return len([g for g in graphs if g.overwrap==False])

    def set_figsize(self, x, y):
        self.figsize = (x, y)

    def make_fig(self):
        self.fig = plt.figure(figsize=self.figsize)
        ax = []
        n = self.num_of_sheets(self.graphs)
        for i, graph in zip(range(len(self.graphs)), self.graphs):
            if not graph.overwrap:
                ax.append(self.fig.add_subplot(1, n, self.num_of_sheets(self.graphs[:i+1])))
                ax[-1].set_title(graph.title)
                ax[-1].set_xlabel(graph.xlabel)
                ax[-1].set_ylabel(graph.ylabel)
            if graph.action == self.Action.TWOVAR_CMAP:
                plt.pcolormesh(graph.input[0], graph.input[1], graph.output, cmap='viridis', shading='auto')
            if graph.action == self.Action.ONEVAR_PLOT:
                if graph.color == "":
                    plt.plot(graph.input, graph.output, label=graph.datalabel)
                else:
                    plt.plot(graph.input, graph.output, label=graph.datalabel, color=graph.color)
            if graph.action == self.Action.ONEVAR_SCATTER:
                if graph.color == "":
                    plt.scatter(graph.input, graph.output, label=graph.datalabel)
                else:
                    plt.scatter(graph.input, graph.output, label=graph.datalabel, color=graph.color)
        plt.tight_layout()
        
    def show(self):
        self.make_fig()
        plt.show()
        
    def save(self, file_name):
        self.make_fig()
        plt.savefig(f"{file_name}.png", format="png")
        
    def recstart(self, file_name, interval=200):
        self.recording = True
        self.ani_interval = interval
        self.ani_filename = file_name
        
    def recend(self):
        ani = animation.ArtistAnimation(self.fig, self.ims, self.ani_interval, blit=True, repeat_delay=100)
        self.recording = False
    
def confirm():
    def f(x, y):
        return np.sin(x) * np.cos(y)
    def g(x, y):
        return x + y
    def h(x):
        return x * x
    vis = Visualize()
    vis.draw_2_variable_func_colormap(f, [0, 5], [0, 5], "sin(x) * cos(y)", "x", "y")
    vis.draw_2_variable_func_colormap(g, [0, 5], [0, 5], "x + y", "x", "y")
    vis.draw_1_variable_func(h, [-2, 2], False, "1d datas", "x ** 2", "x", "y")
    vis.draw_1_variable_data([0, 1, 3, 5], [4, 2, 8, 11], True, "", "random_datas", "x", "y")
    vis.show()
    
def main():
    confirm()
    
if __name__=="__main__":
    main()
    
    
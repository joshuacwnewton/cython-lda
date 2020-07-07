from matplotlib import pyplot


class InteractivePlot(object):

    def __init__(self, xlabel, ylabel):

        pyplot.ion()

        self.xlabel = xlabel
        self.ylabel = ylabel

        self.x = []
        self.y = []

    def update_plot(self, xval, yval):

        self.x.append(xval)
        self.y.append(yval)

        pyplot.clf()

        pyplot.plot(self.y, self.x, 'k')

        pyplot.xlabel(self.xlabel)
        pyplot.ylabel(self.ylabel)

        pyplot.draw()


if __name__ == '__main__':

    import time, math

    plt = InteractivePlot('x', 'y')

    for x in range(100):

        plt.update_plot(x, x * math.sin(0.4 * x))
        time.sleep(.01)

# Mistake created:
# - Line 23: swapped 'x' and 'y' argument positions
#   - Incorrect logic (even if the variables were actually named that way anyway, it makes for bad code readability)

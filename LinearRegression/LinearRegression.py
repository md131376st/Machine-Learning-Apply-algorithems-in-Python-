import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, M,iteration=1500, alpha=0.01):
        """

        :type alpha: double
        """
        self.theta = np.zeros((2, 1))
        self.iteration = iteration
        self.alpha = alpha
        self.historyJ = []
        self.M = M

    def cost(self, x, y):
        h = np.matmul(x, self.theta)
        return sum((1 / (2 * np.size(y))) * (np.power(( h - y), 2)))

    def train(self, x, y):
        for i in range(0,self.iteration):
            temp = np.matmul(x, self.theta) - y
            temp = np.reshape(temp, (1, self.M))
            self.theta = self.theta- np.reshape((self.alpha/self.M) * np.matmul(temp, x), (2,1))
            self.historyJ.append(self.cost(x, y))

    def plotj(self):
        plt.plot(self.historyJ)
        plt.ylabel('Cost Function data')
        plt.show()

    def predict(self, input):
        return self.theta[0] + self.theta[1]*input


if __name__ == '__main__':
    data = np.loadtxt('ex1data1.txt', dtype=float, delimiter=',')
    X = data[:, 0]
    Y = data[:, 1]
    M = np.size(X)
    X = np.append(np.ones((M,1)), np.reshape(X, (M,1)),1)
    Y = np.reshape(data[:, 1], (M,1))
    algorithem = LinearRegression(M)
    algorithem.train(X,Y)
    algorithem.plotj()
    print(algorithem.predict(20))


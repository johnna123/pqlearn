import joblib
import matplotlib.pyplot as plt
import numpy as np

from qlearn import QLearn
from snake_game import SnakeGame


def train(iters, warm_start=False, verbose=False, learning_rate=0.8, gamma=0.8, epsilon=0.2,
          dont_repeat=False):
    if warm_start:
        ai = joblib.load("memory.pkl")
    else:
        ai = QLearn([0, 1, 2, 3])
    ai.learning_rate = learning_rate
    ai.gamma = gamma
    ai.epsilon = epsilon
    ai.verbose = verbose
    ai.no_repeat = dont_repeat
    evals = []
    bu_iter = 100
    for i in range(1, iters + 1):
        game = SnakeGame()
        ai = game.demo_light(ai)
        evals.append(np.sum(np.array([v for v in ai.memory.values()])))
        plt.plot(evals, c="b")
        plt.pause(0.05)
        if not i % bu_iter:
            joblib.dump(ai, "memory.pkl")
    joblib.dump(ai, "memory.pkl")


if __name__ == '__main__':
    train(2000, epsilon=0.8, dont_repeat=True, warm_start=False)

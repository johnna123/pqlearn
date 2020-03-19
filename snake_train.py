import joblib
import matplotlib.pyplot as plt
import numpy as np

from qlearn import QLearn
from snake_game import SnakeGame


def train(iters, warm_start=False, verbose=False, learning_rate=0.8, gamma=0.8, epsilon=0.2,
          dont_repeat=False, name="memory.pkl"):
    """
    QLearn usage example training in the Snake environment

    """
    if warm_start:
        ai = joblib.load(name)
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
        ai = game.demo(ai, light_mode=True)
        evals.append(np.sum(np.array([v for v in ai.memory.values()])))
        plt.plot(evals, c="b")
        plt.pause(0.05)
        if not i % bu_iter:
            joblib.dump(ai, name)
    joblib.dump(ai, name)


if __name__ == '__main__':
    train(2000, epsilon=0.8, dont_repeat=True, warm_start=True, name="ignore_test")

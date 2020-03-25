import os

import numpy


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def dist_reward(old, act):
    """
    Returns a reward given the movement of the snake [-1 -> 1]

        1: moving direct to target

        -1: moving away from target

    :param old: Old distance between snake head and food
    :param act: New distance between snake head and food
    :type old: float
    :type act: float
    :return:
    :rtype: float
    """
    d = act - old
    if -10 <= d <= 10:
        return -d / 10
    else:
        return 0


def dist(a, b):
    """
    Given 2 position vectors returns distance

    :param a: Position vector of point a
    :param b: Position vector of poin b
    :return: |(a-b)|
    :type a: list
    :type b: list
    :rtype: numpy.float64
    """
    a = numpy.array(a)
    b = numpy.array(b)
    return numpy.sqrt(numpy.sum(numpy.power(a - b, 2)))


def state2int(points_earned, distance_reward):
    """
    Helper function returns total reward for snake_game

        ate_food -> reward=2

        no_colition -> reward=distance_reward

        colision -> reward=-2

    :param points_earned: Points earned reported by SnakeGame
    :param distance_reward: Calculated with dp_reward
    :type points_earned: float
    :type distance_reward: float
    :return: Total reward
    :rtype: float or int
    """
    if points_earned == 0:
        return distance_reward
    elif points_earned < 0:
        return -2
    elif points_earned > 0:
        return 2
    return 0

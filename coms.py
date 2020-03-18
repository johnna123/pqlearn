import numpy


def dp_reward(old, act):
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


def state2int(deltP, deltD):
    if deltP == 0:
        return deltD
    elif deltP < 0:
        return -2
    elif deltP > 0:
        return 2
    return 0

import numpy as np


def decibel_to_volts(decibel):
    return 1e-6 * 10 ** (decibel / 20)


def volts_to_decibel(volts):
    return 20 ** np.log10(volts * 1e6)

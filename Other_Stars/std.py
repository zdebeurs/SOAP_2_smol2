import numpy as np
import math as m

# A simple fwhm finder
# flip over ccf, find max, find half max locations, distance between those two locations, divide
def std(x):
    return np.std(x, ddof=1)

#returns fwhm and sigma


if __name__ == '__main__':
    std()
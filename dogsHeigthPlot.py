import numpy as np
import matplotlib.pyplot as plt

# Number of Greyhound and Labrador samples
greyhoundCount = 500
labradorCount = 500

# Array of 500 sample heigths for both types of dogs
# Each high is (fixed heigth(in inches)+- 4(inches))
greyHeigth = 28 + 4*np.random.randn(greyhoundCount)
labHeigth = 24 + 4*np.random.randn(labradorCount)

# Plots a histogram containing the 500 samples of each dog heigth
# Red color: greyhounds, Blue color: labradors
plt.hist([greyHeigth, labHeigth], stacked=True, color=['r', 'b'])
plt.show()
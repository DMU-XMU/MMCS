import numpy as np

array_curl = [1176.2, 2057.4, 1292.6]

array_mmcs = [1960, 2548.8, 2428.8]
sea_array = [919.2, 1979.8,2487.2]
array = array_mmcs
mean = np.mean(sea_array)
std = np.std(sea_array)
print("mean:" + str(mean)+" std:"+str(std))
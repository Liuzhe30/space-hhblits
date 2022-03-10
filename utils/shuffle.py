import numpy as np


x = np.load('/home/panwh/space-hhblits/data/dataset_5A_simple_average_shhm_slidingwindow_x.npy')
y = np.load('/home/panwh/space-hhblits/data/dataset_5A_simple_average_shhm_slidingwindow_y.npy')

seed = 500
np.random.seed(seed)
np.random.shuffle(x)
np.random.seed(seed)
np.random.shuffle(y)

np.save('/home/panwh/space-hhblits/data/dataset_5A_simple_average_shhm_slidingwindow_x.npy',x)
np.save('/home/panwh/space-hhblits/data/dataset_5A_simple_average_shhm_slidingwindow_y.npy',y)

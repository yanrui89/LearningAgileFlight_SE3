import numpy as np
import os
import matplotlib.pyplot as plt

file_path = '/home/tlabstaff/storage/LearningAgileFlight_SE3'
reward_weight = 50.0
gamma = 10.0
alpha = 10.0
beta = 10.0
folder_name = "reward_" + str(reward_weight) + "_gamma_" + str(gamma) + "_beta_" + str(beta) + "_alpha_" + str(alpha)
full_folder_path = os.path.join(file_path, folder_name)

total_iteration = 4

file_name = ['Every_collide',
             'Every_comp1',
             'Every_comp2',
             'Every_delat1',
             'Every_delta2',
             'Every_delta3',
             'Every_delta4',
             'Mean_Reward',
             'Every_reward',
             'Every_totalreward',
             'Every_currreward',
             'Every_path']
# file_name=['Every_reward']

for curr_file in file_name:
    fig, ax = plt.subplots(total_iteration)
    fig.suptitle(curr_file)
    for iter in range(total_iteration):
        full_curr_file = curr_file + str(iter) + '.npy'
        full_path = os.path.join(full_folder_path, full_curr_file)
        print(full_path)

        a = np.load(full_path)
        # for i in range(a.shape[0]):
        #     curr = a[i,:]
        #     test = np.where(np.abs(curr) > 10e5)
        if len(a.shape) > 1:
            avg = np.average(a, axis=1)
        else:
            avg = a
        ax[iter].plot(np.arange(a.shape[0]),avg)
        # ax[iter].set_ylim([-10e5, 10])

    plt.show()
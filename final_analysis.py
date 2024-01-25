import numpy as np
import os

reward_weights = [100.0]
alphas = [10.0, 100.0]
betas = [10.0, 100.0]
gammas = [10.0, 100.0]
ep_betas = [0.000001, 0.00001,0.0001, 0.001]


for reward_weight in reward_weights:
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                for ep_beta in ep_betas:
                    home_dir = os.path.expanduser('~')
                    work_dir = os.path.join(home_dir, "storage/LearningAgileFlight_SE3")
                    load_path = work_dir + "/reward_" + str(reward_weight) + "_gamma_" + str(gamma) + "_beta_" + str(beta) + "_alpha_" + str(alpha) + "_ep_beta_" + str(ep_beta)
                    FILE = load_path + "/Analysis.npy"

                    dict_load = np.load(FILE, allow_pickle=True).item()
                    curr_collide = dict_load["CollideList"]
                    curr_endPt = dict_load["EndPtList"]
                    passthrough = dict_load["PassThroughList"]
                    distWind = dict_load["DistFromWindList"]

                    collide_array = np.array(curr_collide)

                    count = 0
                    accum_distWind = 0
                    accum_endPt = 0

                    for i in range(len(passthrough)):
                        curr_curr_collide = collide_array[i]
                        curr_passthrough = passthrough[i]
                        
                        if curr_curr_collide == 0 and curr_passthrough == True:
                            count += 1
                            accum_distWind += distWind[i]
                            accum_endPt += curr_endPt[i]


                    success_rate = count / len(passthrough)
                    avg_distWind = accum_distWind / count
                    avg_endPt = accum_endPt / count

                    print (f'reward_weight: {reward_weight}, alpha: {alpha}, beta: {beta}, gamma: {gamma}, ep_beta: {ep_beta}, success_rate: {success_rate}, distance from window: {avg_distWind}, distance from end point: {avg_endPt}')



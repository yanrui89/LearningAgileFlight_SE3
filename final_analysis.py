import numpy as np

reward_weights = [100.0]
alphas = [100.0, 10.0]
betas = [10.0, 100.0]
gammas = [10.0, 100.0]


for reward_weight in reward_weights:
    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                work_dir = "/home/tlabstaff/storage/LearningAgileFlight_SE3"
                load_path = work_dir + "/reward_" + str(reward_weight) + "_gamma_" + str(gamma) + "_beta_" + str(beta) + "_alpha_" + str(alpha) 
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

                print (f'reward_weight: {reward_weight}, alpha: {alpha}, beta: {beta}, gamma: {gamma}, success_rate: {success_rate}, distance from window: {avg_distWind}, distance from end point: {avg_endPt}')



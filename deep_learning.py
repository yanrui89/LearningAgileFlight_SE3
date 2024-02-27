## this file is for deep learning

from quad_nn import *
from quad_policy_with_OC import *
from multiprocessing import Process, Array
import sys
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
# initialization


## deep learning
# Hyper-parameters 

num_epochs = 100
batch_size = 100 # 100
learning_rate = 1e-4
num_cores =20 #5


def writeToTensorBoard(writer, tensorboardData, curr_episode):

    reward, full_collision, curr_collision, every_comp1, every_comp2, every_collide, every_curr_delta1, every_curr_delta2, every_curr_delta3, every_curr_delta4, every_path = tensorboardData

    writer.add_scalar(tag='Losses/Reward', scalar_value=reward, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Full Collision', scalar_value=full_collision, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Curr Collision', scalar_value=curr_collision, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Comp1', scalar_value=every_comp1, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Comp2', scalar_value=every_comp2, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Every Collide', scalar_value=every_collide, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Delta 1', scalar_value=every_curr_delta1, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Delta 2', scalar_value=every_curr_delta2, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Delta 3', scalar_value=every_curr_delta3, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Delta 4', scalar_value=every_curr_delta4, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Path', scalar_value=every_path, global_step=curr_episode)


# for multiprocessing, obtain the gradient
def grad(inputs, outputs, gra, reward_weight, alpha, beta, gamma, ep_beta, ep_bool):
    gate_point = np.array([[-inputs[7]/2,0,1],[inputs[7]/2,0,1],[inputs[7]/2,0,-1],[-inputs[7]/2,0,-1]])
    gate1 = gate(gate_point)
    gate_point = gate1.rotate_y_out(inputs[8])

    quad1 = run_quad_withOC(goal_pos=inputs[3:6],ini_r=inputs[0:3].tolist(),ini_q=toQuaternion(inputs[6],[0,0,1]), reward_weight = reward_weight,\
                      ep_beta = ep_beta, alpha = alpha, beta = beta, gamma = gamma)
    quad1.init_obstacle(gate_point.reshape(12))

    gra[:] = quad1.sol_gradient(quad1.ini_state,outputs[0:3],outputs[3:6],outputs[6], ep_bool = ep_bool)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-reward_weight", type=float, default=100.0)
    parser.add_argument("-alpha", type=float, default=100.0)
    parser.add_argument("-beta", type=float, default=10.0)
    parser.add_argument("-gamma", type=float, default=1000.0)
    parser.add_argument("-ep_beta", type=float, default=0.001)
    parser.add_argument("-ep_bool", action="store_false")
    parser.add_argument("-use_gpu", action="store_false")
    parser.add_argument("-num_runs", type=int, default=1)
    args = parser.parse_args()


    home_dir = os.path.expanduser('~')
    work_dir = os.path.join(home_dir, "storage/LearningAgileFlight_SE3")
    FILE = os.path.join(work_dir, "nn_pre.pth")
    save_path = os.path.join(work_dir, "reward_" + str(args.reward_weight) + "_gamma_" + str(args.gamma) + \
                                "_beta_" + str(args.beta) + "_alpha_" + str(args.alpha) + "_ep_beta_" + str(args.ep_beta) + \
                                    "_ep_bool_" + str(int(args.ep_bool)))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print(f"Running with reward weight {args.reward_weight}, alpha {args.alpha} beta {args.beta} gamma {args.gamma} ep_beta {args.ep_beta} and ep_bool {args.ep_bool}")
    for k in range(args.num_runs):

        writer = SummaryWriter(os.path.join(save_path, "run_" + str(k)))

        Every_reward = np.zeros((num_epochs,batch_size))
        Every_fullCollision = np.zeros((num_epochs,batch_size))
        Every_currCollision = np.zeros((num_epochs,batch_size))
        Every_collide = np.zeros((num_epochs,batch_size))
        Every_comp1 = np.zeros((num_epochs,batch_size))
        Every_comp2 = np.zeros((num_epochs,batch_size))
        Every_currDelta1 = np.zeros((num_epochs,batch_size))
        Every_currDelat2 = np.zeros((num_epochs,batch_size))
        Every_currDelta3 = np.zeros((num_epochs,batch_size))
        Every_currDelta4 = np.zeros((num_epochs,batch_size))
        Every_path = np.zeros((num_epochs,batch_size))
        training_data = []


        best_reward = -10000

        model = torch.load(FILE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        Iteration = []
        Mean_r = []
        for epoch in range(num_epochs):
            evalue = 0
            Iteration += [epoch+1]
            for i in range(int(batch_size/num_cores)):
                n_inputs = []
                n_outputs = []
                n_out = []
                n_gra = []
                n_process = []
                for _ in range(num_cores):
                # sample
                    inputs = nn_sample()
                # forward pass
                    outputs = model(inputs)
                    out = outputs.data.numpy()
                # create shared variables
                    gra = Array('d',np.zeros(17))
                # collection
                    n_inputs.append(inputs)
                    n_outputs.append(outputs)
                    n_out.append(out)
                    n_gra.append(gra)

                #calculate gradient and loss
                for j in range(num_cores):
                    p = Process(target=grad,args=(n_inputs[j],n_out[j],n_gra[j], args.reward_weight, args.alpha, args.beta, args.gamma, args.ep_beta, args.ep_bool))
                    p.start()
                    n_process.append(p)
        
                for process in n_process:
                    process.join()

                # Backward and optimize
                for j in range(num_cores):                
                    outputs = model(n_inputs[j])
                    loss = model.myloss(outputs,n_gra[j][0:7])        

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    evalue += n_gra[j][7]
                    Every_reward[epoch,j+num_cores*i]=n_gra[j][7]
                    Every_currCollision[epoch,j+num_cores*i]=n_gra[j][9]
                    Every_comp1[epoch,j+num_cores*i]=n_gra[j][9]
                    Every_comp2[epoch,j+num_cores*i]=n_gra[j][10]
                    Every_collide[epoch,j+num_cores*i]=n_gra[j][11]
                    Every_currDelta1[epoch,j+num_cores*i]=n_gra[j][12]
                    Every_currDelat2[epoch,j+num_cores*i]=n_gra[j][13]
                    Every_currDelta3[epoch,j+num_cores*i]=n_gra[j][14]
                    Every_currDelta4[epoch,j+num_cores*i]=n_gra[j][15]
                    Every_path[epoch,j+num_cores*i]=n_gra[j][16]



                training_data = [Every_reward[epoch,num_cores*i:j+1 +num_cores*i].mean(),
                                Every_fullCollision[epoch,num_cores*i:j+1+num_cores*i].mean(),
                                Every_currCollision[epoch,num_cores*i:j+1+num_cores*i].mean(),
                                Every_comp1[epoch,num_cores*i:j+1+num_cores*i].mean(),
                                Every_comp2[epoch,num_cores*i:j+1+num_cores*i].mean(),
                                Every_collide[epoch,num_cores*i:j+1+num_cores*i].mean(),
                                Every_currDelta1[epoch,num_cores*i:j+1+num_cores*i].mean(),
                                Every_currDelat2[epoch,num_cores*i:j+1+num_cores*i].mean(),
                                Every_currDelta3[epoch,num_cores*i:j+1+num_cores*i].mean(),
                                Every_currDelta4[epoch,num_cores*i:j+1+num_cores*i].mean(),
                                Every_path[epoch,num_cores*i:j+1+num_cores*i].mean()]

                writeToTensorBoard(writer, training_data, i+epoch*int(batch_size/num_cores))

                if (i+1)%1 == 0:
                    print (f'Iterate: {k}, Epoch [{epoch+1}/{num_epochs}], Step [{(i+1)*num_cores}/{batch_size}], Reward: {n_gra[0][7]:.4f}')
            # change state
            mean_reward = evalue/batch_size # evalue/int(batch_size/num_cores)
            if mean_reward > best_reward:
                torch.save(model, save_path + "/nn_deep2_best")
            Mean_r += [mean_reward]
            print('evaluation: ',mean_reward)
            np.save(save_path + '/Iteration',Iteration)
            np.save(save_path + '/Mean_Reward'+str(k),Mean_r)
            np.save(save_path + '/Every_reward'+str(k),Every_reward)
            np.save(save_path + '/Every_totalreward'+str(k),Every_fullCollision)
            np.save(save_path + '/Every_currreward'+str(k),Every_currCollision)
            np.save(save_path + '/Every_collide'+str(k),Every_collide)
            np.save(save_path + '/Every_comp1'+str(k),Every_comp1)
            np.save(save_path + '/Every_comp2'+str(k),Every_comp2)
            np.save(save_path + '/Every_delat1'+str(k),Every_currDelta1)
            np.save(save_path + '/Every_delta2'+str(k),Every_currDelat2)
            np.save(save_path + '/Every_delta3'+str(k),Every_currDelta3)
            np.save(save_path + '/Every_delta4'+str(k),Every_currDelta4)
            np.save(save_path + '/Every_path'+str(k),Every_path)
        torch.save(model, save_path + "/nn_deep2_"+str(k))
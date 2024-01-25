## this file is for deep learning

from quad_nn import *
from quad_model import *
from quad_policy import *
from multiprocessing import Process, Array
import sys
import os
# initialization


## deep learning
# Hyper-parameters 

num_epochs = 100
batch_size = 100 # 100
learning_rate = 1e-4
num_cores =20 #5

home_dir = os.path.expanduser('~')
work_dir = os.path.join(home_dir, "storage/LearningAgileFlight_SE3")
FILE = work_dir + "/nn_pre.pth"
model = torch.load(FILE)
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


# for multiprocessing, obtain the gradient
def grad(inputs, outputs, gra, reward_weight, alpha, beta, gamma, ep_beta, ep_bool):
    gate_point = np.array([[-inputs[7]/2,0,1],[inputs[7]/2,0,1],[inputs[7]/2,0,-1],[-inputs[7]/2,0,-1]])
    gate1 = gate(gate_point)
    gate_point = gate1.rotate_y_out(inputs[8])

    quad1 = run_quad(goal_pos=inputs[3:6],ini_r=inputs[0:3].tolist(),ini_q=toQuaternion(inputs[6],[0,0,1]), reward_weight = reward_weight, ep_beta = ep_beta)
    quad1.init_obstacle(gate_point.reshape(12), alpha = alpha, beta=beta, gamma = gamma)
    quad1.setObstacle()

    gra[:] = quad1.sol_gradient(quad1.ini_state,outputs[0:3],outputs[3:6],outputs[6], ep_bool = ep_bool)

if __name__ == '__main__':
    reward_weight = float(sys.argv[1])
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    gamma = float(sys.argv[4])
    ep_beta = float(sys.argv[5])
    ep_bool_str = float(sys.argv[6])
    if ep_bool_str == 1:
        ep_bool = 1
    else:
        ep_bool = 0
    # reward_weight = 100.0
    # alpha = 10.0
    # beta = 10.0
    # gamma = 10.0
    # ep_beta = 0.001
    # ep_bool = 1
    print(f"Running with reward weight {reward_weight}, alpha {alpha} beta {beta} gamma {gamma} ep_beta {ep_beta} and ep_bool {ep_bool}")
    for k in range(1):
        best_reward = -10000
        home_dir = os.path.expanduser('~')
        work_dir = os.path.join(home_dir, "storage/LearningAgileFlight_SE3")
        FILE = work_dir + "/nn_pre.pth"
        save_path = work_dir + "/reward_" + str(reward_weight) + "_gamma_" + str(gamma) + "_beta_" + str(beta) + "_alpha_" + str(alpha) + "_ep_beta_" + str(ep_beta) + "_ep_bool_" + str(ep_bool)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        model = torch.load(FILE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        Iteration = []
        Mean_r = []
        for epoch in range(num_epochs):
        #move = gate1.plane_move()
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
                    gra = Array('d',np.zeros(18))
                # collection
                    n_inputs.append(inputs)
                    n_outputs.append(outputs)
                    n_out.append(out)
                    n_gra.append(gra)

                #calculate gradient and loss
                for j in range(num_cores):
                    p = Process(target=grad,args=(n_inputs[j],n_out[j],n_gra[j], reward_weight, alpha, beta, gamma, ep_beta, ep_bool))
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
                    Every_fullCollision[epoch,j+num_cores*i]=n_gra[j][8]
                    Every_currCollision[epoch,j+num_cores*i]=n_gra[j][9]
                    Every_comp1[epoch,j+num_cores*i]=n_gra[j][10]
                    Every_comp2[epoch,j+num_cores*i]=n_gra[j][11]
                    Every_collide[epoch,j+num_cores*i]=n_gra[j][12]
                    Every_currDelta1[epoch,j+num_cores*i]=n_gra[j][13]
                    Every_currDelat2[epoch,j+num_cores*i]=n_gra[j][14]
                    Every_currDelta3[epoch,j+num_cores*i]=n_gra[j][15]
                    Every_currDelta4[epoch,j+num_cores*i]=n_gra[j][16]
                    Every_path[epoch,j+num_cores*i]=n_gra[j][17]

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
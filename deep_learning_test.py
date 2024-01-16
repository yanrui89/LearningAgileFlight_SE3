## this file is for deep learning

from quad_nn import *
from quad_model import *
from quad_policy import *
from multiprocessing import Process, Array
import sys
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# initialization


## deep learning
# Hyper-parameters 

num_epochs = 200
batch_size = 1 # 100
learning_rate = 1e-4
num_cores =1 #5

work_dir = "/home/tlabstaff/storage/LearningAgileFlight_SE3"
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
def grad(inputs, outputs, gra, reward_weight, alpha, beta, gammma):
    gate_point = np.array([[-inputs[7]/2,0,1],[inputs[7]/2,0,1],[inputs[7]/2,0,-1],[-inputs[7]/2,0,-1]])
    gate1 = gate(gate_point)
    gate_point = gate1.rotate_y_out(inputs[8])

    quad1 = run_quad(goal_pos=inputs[3:6],ini_r=inputs[0:3].tolist(),ini_q=toQuaternion(inputs[6],[0,0,1]), reward_weight = reward_weight)
    quad1.init_obstacle(gate_point.reshape(12), alpha = alpha, beta=beta, gamma = gamma)

    gra[:], traj = quad1.sol_test(quad1.ini_state,outputs[0:3],outputs[3:6],outputs[6])
    trav_t = outputs[6]

    return gra, traj, trav_t, gate_point, inputs[3:6], gra[5]

if __name__ == '__main__':
    reward_weight = float(sys.argv[1])
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    gamma = float(sys.argv[4])
    # reward_weight = 100.0
    # alpha = 100.0
    # beta = 10.0
    # gamma = 10.0
    print(f"Loading with reward weight {reward_weight}, alpha {alpha} beta {beta} and gamma {gamma}")
    count = 0
    for k in range(1):
        work_dir = "/home/tlabstaff/storage/LearningAgileFlight_SE3"
        load_path = work_dir + "/reward_" + str(reward_weight) + "_gamma_" + str(gamma) + "_beta_" + str(beta) + "_alpha_" + str(alpha) 
        FILE = load_path + "/nn_deep2_" + str(k)
        if not os.path.isdir(load_path):
            raise Exception('load path is unavailable')
        model = torch.load(FILE)
        for i in model.parameters():
            print(i)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        Iteration = []
        CollideList = []
        EndPtList = []
        PassThroughList = []
        DistFromWindList = []

        Mean_r = []
        model.eval()
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
                    gra = np.zeros(11)
                # collection
                    n_inputs.append(inputs)
                    n_outputs.append(outputs)
                    n_out.append(out)
                    # n_gra.append(gra)

                #Test time
                    Gra, Traj, Trav_t, GatePoint, GoalPos, collide = grad(inputs, out, gra, reward_weight, alpha, beta, gamma)
                    GatePoints = np.vstack((GatePoint, GatePoint[0,:]))

                    trav_idx = Trav_t / 0.1
                    pre_idx = int(np.floor(trav_idx))
                    intersect = Traj[pre_idx]

                    # print(f"SHOULD PASS THROUGH THE WINDOW HERE! {intersect}")

                    # fig = plt.figure()
                    # ax = fig.add_subplot(projection='3d')
                    # ax.scatter(Traj[:,0], Traj[:,1], Traj[:,2])
                    # ax.plot3D(GatePoints[:,0], GatePoints[:,1], GatePoints[:,2])
                    # ax.scatter(intersect[0],intersect[1], intersect[2], c="red")
                    # ax.scatter(GoalPos[0],GoalPos[1], GoalPos[2], c="green")
                    # ax.set_xlabel('X Label')
                    # ax.set_ylabel('Y Label')
                    # ax.set_zlabel('Z Label')

                    # plt.show()

                    #Find how far the end point is from the goal
                    delta_dist = np.linalg.norm(GoalPos - Traj[-1,:])
                    Gate2D = np.vstack((GatePoint[:,0],GatePoint[:,2])).transpose()
                    polygon = Polygon([Gate2D[0,:], Gate2D[1,:], Gate2D[2,:], Gate2D[3,:]])
                    point = Point(intersect[0], intersect[1])
                    chk = polygon.contains(point)

                    delta_vect1 = GatePoint[1,:] - GatePoint[0,:]
                    delta_vect1 = delta_vect1 / np.linalg.norm(delta_vect1)
                    delta_vect2 = GatePoint[2,:] - GatePoint[0,:]
                    delta_vect2 = delta_vect2 / np.linalg.norm(delta_vect2)
                    norm_vector = np.cross(delta_vect1, delta_vect2)
                    norm_vector = norm_vector / np.linalg.norm(norm_vector)

                    delta_vect3 = intersect - GatePoint[0,:]
                    dist = np.abs(np.dot(delta_vect3, norm_vector))

                    print (f'Test: {count}, Entered Window: {chk}, EndPt Distance: {delta_dist}, Distance From Window: {dist}, Collide with Window: {collide}')
                    CollideList.append(collide)
                    EndPtList.append(delta_dist)
                    PassThroughList.append(chk)
                    DistFromWindList.append(dist)

                    n_gra.append(Gra)

    dict_save = {}
    dict_save["CollideList"] = CollideList
    dict_save["EndPtList"] = EndPtList
    dict_save["PassThroughList"] = PassThroughList
    dict_save["DistFromWindList"]= DistFromWindList
    save_file = load_path + '/Analysis.npy'

    np.save(save_file, dict_save )
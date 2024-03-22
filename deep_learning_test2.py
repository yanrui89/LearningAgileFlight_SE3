## this file is for deep learning

from quad_nn import *
from quad_policy_with_OC import *
from multiprocessing import Process, Array
import sys
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
# initialization


## deep learning
# Hyper-parameters 

num_epochs = 200
batch_size = 1 # 100
learning_rate = 1e-4
num_cores =1 #5


# for multiprocessing, obtain the gradient
def grad(inputs, outputs, gra, reward_weight, alpha, beta, gamma, ep_beta, ep_bool):
    gate_point = np.array([[-inputs[7]/2,0,1],[inputs[7]/2,0,1],[inputs[7]/2,0,-1],[-inputs[7]/2,0,-1]])
    gate1 = gate(gate_point)
    gate_point = gate1.rotate_y_out(inputs[8])

    quad1 = run_quad_withOC(goal_pos=inputs[3:6],ini_r=inputs[0:3].tolist(),ini_q=toQuaternion(inputs[6],[0,0,1]), reward_weight = reward_weight,\
                      ep_beta = ep_beta, alpha = alpha, beta = beta, gamma = gamma)
    quad1.init_obstacle(gate_point.reshape(12))

    a, traj_pos, traj_quat = quad1.sol_test(quad1.ini_state,outputs[0:3],outputs[3:6],outputs[6])
    trav_t = outputs[6]

    return a, traj_pos, traj_quat, trav_t, gate_point, inputs[3:6], a[4]

def plot_pos(Traj, t, tra_pos, tra_quat, GatePoint, goal_pos, traj_quat, tra_ang=None):
    point1 = GatePoint[0,:]
    point2 = GatePoint[1,:]
    point3 = GatePoint[2,:]
    point4 = GatePoint[3,:]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    GatePoints = np.array([point1, point2, point3, point4, point1])
    ax.scatter(Traj[:,0], Traj[:,1], Traj[:,2])
    ax.plot3D(GatePoints[:,0], GatePoints[:,1], GatePoints[:,2])
    trav_idx = t / 0.1
    pre_idx = int(np.floor(trav_idx))
    intersect = Traj[pre_idx]

    ax.scatter(intersect[0],intersect[1], intersect[2], c="red")
    ax.scatter(goal_pos[0],goal_pos[1], goal_pos[2], c="green")
    ax.scatter(tra_pos[0],tra_pos[1], tra_pos[2], c="green")

    tt = int(round(t *10))
    r = R.from_quat(np.array([traj_quat[tt][1],traj_quat[tt][2],traj_quat[tt][3],traj_quat[tt][0]]) )
    new_z = np.matmul(r.as_matrix(), np.array([0,0,1]))
    new_y = np.matmul(r.as_matrix(), np.array([0,1,0]))
    new_x = np.matmul(r.as_matrix(), np.array([1,0,0]))

    new_z_pt = intersect + new_z

    ax.plot(xs=[intersect[0], new_z_pt[0]], ys=[intersect[1],new_z_pt[1]],zs=[intersect[0],new_z_pt[1]] , c='b')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-reward_weight", type=float, default=100.0)
    parser.add_argument("-alpha", type=float, default=100.0)
    parser.add_argument("-beta", type=float, default=10.0)
    parser.add_argument("-gamma", type=float, default=1000.0)
    parser.add_argument("-ep_beta", type=float, default=0.001)
    parser.add_argument("-ep_bool", action="store_true")
    parser.add_argument("-use_gpu", action="store_false")
    parser.add_argument("-num_runs", type=int, default=1)
    args = parser.parse_args()


    home_dir = os.path.expanduser('~')
    work_dir = os.path.join(home_dir, "storage/LearningAgileFlight_SE3")
    
    load_path = os.path.join(work_dir, "reward_" + str(args.reward_weight) + "_gamma_" + str(args.gamma) + \
                                "_beta_" + str(args.beta) + "_alpha_" + str(args.alpha) + "_ep_beta_" + str(args.ep_beta) + \
                                    "_ep_bool_" + str(int(args.ep_bool)))
    FILE = load_path + "/nn_deep2_0"
    assert os.path.isfile(FILE), 'file does not exist'

    print(f"Running with reward weight {args.reward_weight}, alpha {args.alpha} beta {args.beta} gamma {args.gamma} ep_beta {args.ep_beta} and ep_bool {args.ep_bool}")
    count = 0
    for k in range(args.num_runs):
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

                Gra, Traj, Traj_quat, Trav_t, GatePoint, GoalPos, collide = grad(inputs, out, gra, args.reward_weight, args.alpha, args.beta, args.gamma, args.ep_beta, args.ep_bool)
                GatePoints = np.vstack((GatePoint, GatePoint[0,:]))

                trav_idx = Trav_t / 0.1
                pre_idx = int(np.floor(trav_idx))
                intersect = Traj[pre_idx]

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


    dict_save = {}
    dict_save["CollideList"] = CollideList
    dict_save["EndPtList"] = EndPtList
    dict_save["PassThroughList"] = PassThroughList
    dict_save["DistFromWindList"]= DistFromWindList
    save_file = load_path + '/Analysis.npy'
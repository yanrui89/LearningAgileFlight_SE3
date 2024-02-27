## this file is a package for policy search for quadrotor

from quad_model_with_OC import *
from math import cos, pi, sin, sqrt, tan
from casadi import *
import scipy.io as sio
import numpy as np
from solid_geometry import *
def Rd2Rp(tra_ang):
    theta = 2*math.atan(magni(tra_ang))
    vector = norm(tra_ang+np.array([1e-8,0,0]))
    return [theta,vector]

class run_quad_withOC:
    def __init__(self, goal_pos = [0, 8, 0], goal_atti = [0,[1,0,0]], ini_r=[0,-8,0]\
            ,ini_v_I = [0.0, 0.0, 0.0], ini_q = toQuaternion(0.0,[3,3,5]),horizon = 50, 
            reward_weight = 1000, ep_beta = 0.01, alpha = 10.0, beta= 10.0, gamma=10.0):
        ## drone definition
        self.winglen = 1.5
        # goal definition
        self.goal_pos = goal_pos
        self.goal_atti = goal_atti 
        # initial state definition
        if type(ini_r) is not list:
            ini_r = ini_r.tolist()
        self.ini_r = ini_r
        self.ini_v_I = ini_v_I 
        self.ini_q = ini_q
        self.ini_w =  [0.0, 0.0, 0.0]
        self.ini_state = self.ini_r + self.ini_v_I + self.ini_q + self.ini_w
        # set horizon definition
        self.horizon = horizon
        self.dt = 0.1

        #reward weights definition
        self.reward_weight = reward_weight
        self.ep_beta = ep_beta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # --------------------------- create model1 ----------------------------------------
        #bounds definition
        sc   = 1e20
        wc   = pi/2 #pi
        tw   = 1.22
        t2w  = 2
        
        self.uav1 = QuadrotorOC()
        jx, jy, jz = 0.0023, 0.0023, 0.004
        self.uav1.initDyn(Jx=0.0023,Jy=0.0023,Jz=0.004,mass=0.5,l=0.35,c=0.0245) # hb quadrotor

        self.uav1.setStateVariable(self.uav1.X,state_lb=[-sc,-sc,-sc,-sc,-sc,-sc,-sc,-sc,-sc,-sc,-wc,-wc,-wc],state_ub=[sc,sc,sc,sc,sc,sc,sc,sc,sc,sc,wc,wc,wc])
        self.uav1.setControlVariable(self.uav1.U,control_lb=[0,0,0,0],control_ub= [t2w*tw,t2w*tw,t2w*tw,t2w*tw])
        self.uav1.initCost(wrt=5,wqt=80,wthrust=0.1,wrf=100,wvf=5,wqf=0,wwf=3, weptl = self.beta, goal_pos=self.goal_pos) # wthrust = 0.1
         # thrust-to-weight = 4:1
        self.uav1.setDyn(self.dt)

    # define function
    # initialize the narrow window
    def init_obstacle(self,gate_point):
        self.point1 = gate_point[0:3]
        self.point2 = gate_point[3:6]
        self.point3 = gate_point[6:9]
        self.point4 = gate_point[9:12]        
        # self.obstacle1 = obstacle(self.point1,self.point2,self.point3,self.point4)
        self.uav1.setObstacle(self.point1, self.point2, self.point3, self.point4, self.gamma, self.alpha, self.beta, self.reward_weight, self.winglen/2)

    def objective( self,ini_state = None,tra_pos=None,tra_ang=None,t = 3, Ulast = None, incl_ep = 0, verbose = True):
        if ini_state is None:
            ini_state = self.ini_state
        t = round(t,1)
        tra_atti = Rd2Rp(tra_ang)
        # define traverse cost
        self.uav1.init_TraCost(tra_pos,tra_atti, t)
        if incl_ep == 1:
            #include the ep loss in the loss function
            self.uav1.init_eptrajloss()
            self.uav1.init_eppathloss(self.reward_weight)
        # obtain solution of trajectory
        sol1 = self.uav1.ocSolver(ini_state=ini_state ,horizon=self.horizon,dt=self.dt, Ulast=Ulast, incl_ep = incl_ep)
        state_traj1 = sol1['state_traj_opt']
        traj = self.uav1.get_quadrotor_position(wing_len = self.winglen, state_traj = state_traj1)
        traj_quaternion = sol1['state_traj_opt'][:, 6:10]
        traj_pos = sol1['state_traj_opt'][:, :3]
        # calculate trajectory reward
        self.path = 0
        ## detect whether there is detection
        reward_data = self.uav1.collis_det(traj_pos, traj_quaternion, t, verbose)
        for p in range(4):
            self.path += np.dot(traj[self.horizon-1-p,0:3]-self.goal_pos, traj[self.horizon-1-p,0:3]-self.goal_pos)
        reward = 1 * reward_data[0] - (self.reward_weight * self.path)
        if verbose == True:
            return [reward] + list(reward_data) + [self.path], traj, traj_pos, traj_quaternion
        else:
            return reward, traj, traj_pos, traj_quaternion
    # --------------------------- solution and learning----------------------------------------
    ##solution and demo
    def sol_gradient(self,ini_state = None,tra_pos =None,tra_ang=None,t=None,Ulast=None, ep_bool = 0):
        tra_ang = np.array(tra_ang)
        tra_pos = np.array(tra_pos)
        reward_data, traj_free,traj_pos_free, traj_quat_free = self.objective(ini_state,tra_pos,tra_ang,t)
        print(t)
        j, curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4, path = reward_data

        if ep_bool == 1:
            trav_idx = t / 0.1
            pre_idx = int(np.floor(trav_idx))
            rp_free, ra_free, rt_free = self.grad_computation(traj_pos_free, traj_quat_free, pre_idx, tra_pos,tra_ang,t)
            rew, traj_perturbed, traj_pos_perturbed, traj_quat_perturbed = self.objective (ini_state,tra_pos,tra_ang,t, incl_ep = 1, verbose = True)
            rp_perturbed, ra_perturbed, rt_perturbed = self.grad_computation(traj_pos_perturbed, traj_quat_perturbed, pre_idx, tra_pos,tra_ang,t)
            ## fixed perturbation to calculate the gradient
            
            drdx = np.clip((rp_perturbed[0] - rp_free[0]) / self.ep_beta, -0.5, 0.5)*0.1
            drdy = np.clip((rp_perturbed[1] - rp_free[1]) / self.ep_beta, -0.5,0.5)*0.1
            drdz = np.clip((rp_perturbed[2] - rp_free[2]) / self.ep_beta, -0.5,0.5)*0.1
            drda = np.clip((ra_perturbed[0] - ra_free[0]) / self.ep_beta, -0.5,0.5)*0.1
            drdb = np.clip((ra_perturbed[1] - ra_free[1]) / self.ep_beta, -0.5,0.5)*0.1
            drdc = np.clip((ra_perturbed[2] - ra_free[2]) / self.ep_beta, -0.5,0.5)*0.1
            drdt = np.clip((rt_perturbed - rt_free) / self.ep_beta, -0.5,0.5)*0.1


            temp1 = Rd2Rp(tra_ang)
            temp2 = toQuaternion(temp1[0],temp1[1])

            print(drdt)

            # delta = 1e-3

            # drdx1 = self.objective(ini_state,tra_pos+[delta*drdx,0,0],tra_ang, t,Ulast, verbose=True)
            # drdy1 = self.objective(ini_state,tra_pos+[0,delta*drdy,0],tra_ang, t,Ulast, verbose=True)
            # drdz1 = self.objective(ini_state,tra_pos+[0,0,delta*drdz],tra_ang, t,Ulast, verbose=True)


        else:
            delta = 1e-3
            drdx = np.clip(self.objective(ini_state,tra_pos+[delta,0,0],tra_ang, t,Ulast, verbose=False)[0] - j,-0.5,0.5)*0.1
            drdy = np.clip(self.objective(ini_state,tra_pos+[0,delta,0],tra_ang, t,Ulast, verbose=False)[0] - j,-0.5,0.5)*0.1
            drdz = np.clip(self.objective(ini_state,tra_pos+[0,0,delta],tra_ang, t,Ulast, verbose=False)[0] - j,-0.5,0.5)*0.1
            drda = np.clip(self.objective(ini_state,tra_pos,tra_ang+[delta,0,0], t,Ulast, verbose=False)[0] - j,-0.5,0.5)*(1/(500*tra_ang[0]**2+5))
            drdb = np.clip(self.objective(ini_state,tra_pos,tra_ang+[0,delta,0], t,Ulast, verbose=False)[0] - j,-0.5,0.5)*(1/(500*tra_ang[1]**2+5))
            drdc = np.clip(self.objective(ini_state,tra_pos,tra_ang+[0,0,delta], t,Ulast, verbose=False)[0] - j,-0.5,0.5)*(1/(500*tra_ang[2]**2+5))

            drdt =0
            if((self.objective(ini_state,tra_pos,tra_ang,t-0.1, verbose=False)[0]-j)>2):
                drdt = -0.05
            if((self.objective(ini_state,tra_pos,tra_ang,t+0.1, verbose = False)[0]-j)>2):
                drdt = 0.05
        ## return gradient and reward (for deep learning)
        return np.array([drdx,drdy,drdz,drda,drdb,drdc,drdt,j, curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4, path])
    
    def set_axes_equal(ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    def grad_computation(self, traj, traj_q, idx, tra_pos,tra_ang,t):

        self.r_I_gra = SX.sym('rxg', 3,1)

        self.q_gra = SX.sym('q_g', 4, 1)

        self.t_r_I_gra = SX.sym('trxg', 3,1)

        self.t_ang_gra = SX.sym('tag', 3, 1)

        self.traversal_time = SX.sym('ttime', 1, 1)
        

        for k in range(int(self.horizon)):
            real_tra_pos = traj[k]
            real_tra_q = traj_q[k]
            self.uav1.total_GraCost(self.r_I_gra,self.t_r_I_gra, self.t_ang_gra, self.q_gra, self.traversal_time, k, self.dt )
            drp = casadi.jacobian(self.uav1.tra_cost_g, self.t_r_I_gra)
            dra = casadi.jacobian(self.uav1.tra_cost_g, self.t_ang_gra)
            drt = casadi.jacobian(self.uav1.tra_cost_g, self.traversal_time)

            f_p = casadi.Function('gra_p',[self.r_I_gra, self.q_gra, self.t_r_I_gra, self.t_ang_gra, self.traversal_time],[drp],['rxg','q_g', 'trxg', 'tag', 'ttime'],['rwp'])
            f_a = casadi.Function('gra_a',[self.r_I_gra, self.q_gra, self.t_r_I_gra, self.t_ang_gra, self.traversal_time],[dra],['rxg','q_g', 'trxg', 'tag', 'ttime'],['rwa'])
            f_t = casadi.Function('gra_t',[self.r_I_gra, self.q_gra, self.t_r_I_gra, self.t_ang_gra, self.traversal_time],[drt],['rxg','q_g', 'trxg', 'tag', 'ttime'],['rwt'])

            real_tra_pos_casadi = casadi.SX([real_tra_pos[0],real_tra_pos[1],real_tra_pos[2]])
            real_tra_q_casadi = casadi.SX([real_tra_q[0],real_tra_q[1],real_tra_q[2], real_tra_q[3]])
            nn_tra_pos_casadi = casadi.SX([tra_pos[0],tra_pos[1],tra_pos[2]])
            nn_tra_q_casadi = casadi.SX([tra_ang[0], tra_ang[1], tra_ang[2]])
            rp = f_p(real_tra_pos_casadi, real_tra_q_casadi, nn_tra_pos_casadi, nn_tra_q_casadi, t)
            ra = f_a(real_tra_pos_casadi, real_tra_q_casadi, nn_tra_pos_casadi, nn_tra_q_casadi, t)
            rt = f_t(real_tra_pos_casadi, real_tra_q_casadi, nn_tra_pos_casadi, nn_tra_q_casadi, t)
            # self.uav1.total_GraCost_time_grad(real_tra_pos,real_tra_q, tra_pos, tra_ang, self.traversal_time, k, self.dt )
            # self.uav1.total_GraCost_time_grad(real_tra_pos,real_tra_q, tra_pos, tra_ang, t, k, self.dt )
            
            if k == 0:
                rp_total = rp
                ra_total = ra
                rt_total = rt
            else:
                rp_total += rp
                ra_total += ra
                rt_total += rt
        return rp_total, ra_total, rt_total


    def sol_test(self,ini_state = None,tra_pos =None,tra_ang=None,t=None,Ulast=None):
        tra_ang = np.array(tra_ang)
        tra_pos = np.array(tra_pos)
        reward_data, traj_test,traj_pos_test, traj_quat_test = self.objective (ini_state,tra_pos,tra_ang,t)
        j, curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4, path = reward_data

        return np.array([j, curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4, path]), traj_test
    
    def plot(self, Traj, t, tra_pos, tra_quat, tra_ang=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        GatePoints = np.array([self.point1, self.point2, self.point3, self.point4, self.point1])
        ax.scatter(Traj[:,0], Traj[:,1], Traj[:,2])
        ax.plot3D(GatePoints[:,0], GatePoints[:,1], GatePoints[:,2])
        trav_idx = t / 0.1
        pre_idx = int(np.floor(trav_idx))
        intersect = Traj[pre_idx]
        pts = self.uav1.collis_det_ep_trial(tra_quat[pre_idx])
        pts_array = np.array(pts)
        pts_array += Traj[pre_idx]

        if tra_ang != None:
            pts = self.uav1.collis_det_ep_trial(tra_ang)
            pts_array1 = np.array(pts)
            pts_array1 += Traj[pre_idx]
            ax.scatter(pts_array1[:,0],pts_array1[:,1], pts_array1[:,2], c="blue")

        ax.scatter(intersect[0],intersect[1], intersect[2], c="red")
        ax.scatter(pts_array[:,0],pts_array[:,1], pts_array[:,2], c="yellow")
        ax.scatter(self.goal_pos[0],self.goal_pos[1], self.goal_pos[2], c="green")
        ax.scatter(tra_pos[0],tra_pos[1], tra_pos[2], c="green")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        self.set_axes_equal(ax)
    
    # def optimize(self, t):
    #     tra_pos = self.obstacle1.centroid
    #     tra_posx = self.obstacle1.centroid[0]
    #     tra_posy = self.obstacle1.centroid[1]
    #     tra_posz = self.obstacle1.centroid[2]
    #     tra_a = 0
    #     tra_b = 0
    #     tra_c = 0
    #     tra_ang = np.array([tra_a,tra_b,tra_c])
    #     ## fixed perturbation to calculate the gradient
    #     for k in range(200):
    #         j = self.objective (tra_pos,tra_ang,t)
    #         drdx = np.clip(self.objective(tra_pos+[0.001,0,0],tra_ang=tra_ang, t=t) - j,-0.5,0.5)
    #         drdy = np.clip(self.objective(tra_pos+[0,0.001,0],tra_ang=tra_ang, t=t) - j,-0.5,0.5)
    #         drdz = np.clip(self.objective(tra_pos+[0,0,0.001],tra_ang=tra_ang, t=t) - j,-0.5,0.5)
    #         drda = np.clip(self.objective(tra_pos,tra_ang=tra_ang+[0.001,0,0], t=t) - j,-0.5,0.5)
    #         drdb = np.clip(self.objective(tra_pos,tra_ang=tra_ang+[0,0.001,0], t=t) - j,-0.5,0.5)
    #         drdc = np.clip(self.objective(tra_pos,tra_ang=tra_ang+[0,0,0.001], t=t) - j,-0.5,0.5)
    #         #drdt = np.clip(self.objective(tra_pos,tra_ang,t-0.1)-j,-10,10)
    #         # update
    #         tra_posx += 0.1*drdx
    #         tra_posy += 0.1*drdy
    #         tra_posz += 0.1*drdz
    #         tra_a += (1/(500*tra_a**2+5))*drda
    #         tra_b += (1/(500*tra_b**2+5))*drdb
    #         tra_c += (1/(500*tra_c**2+5))*drdc
    #         if((self.objective(tra_pos,tra_ang,t-0.1)-j)>2):
    #             t = t-0.1
    #         if((self.objective(tra_pos,tra_ang,t+0.1)-j)>2):
    #             t = t+0.1
    #         t = round(t,1)
    #         tra_pos = np.array([tra_posx,tra_posy,tra_posz])
    #         tra_ang = np.array([tra_a,tra_b,tra_c])
    #         ## display the process
    #         print(str(j)+str('  ')+str(tra_pos)+str('  ')+str(tra_ang)+str('  ')+str(t)+str('  ')+str(k))
    #     return [t,tra_posx,tra_posy,tra_posz,tra_a, tra_b,tra_c, j,self.collision,self.path]

    # ## use random perturbations to calculate the gradient and update(not recommonded)
    # def LSFD(self,t):
    #     tra_posx = self.obstacle1.centroid[0]
    #     tra_posy = self.obstacle1.centroid[1]
    #     tra_posz = self.obstacle1.centroid[2]
    #     tra_a = 0
    #     tra_b = 0
    #     tra_c = 0
    #     current_para = np.array([tra_posx,tra_posy,tra_posz,tra_a,tra_b,tra_c])
    #     lr = np.array([2e-4,2e-4,2e-4,5e-5,5e-5,5e-5])
    #     for k in range(50):
    #         j = self.objective(current_para[0:3],current_para[3:6],t)
    #         # calculate derivatives
    #         c = []
    #         f = []
    #         for i in range(24):
    #             dx = sample(0.001)
    #             dr = self.objective (current_para[0:3]+dx[0:3],current_para[3:6]+dx[3:6],t)-j
    #             c += [dx]
    #             f += [dr]
    #         # update
    #         cm = np.array(c)
    #         fm = np.array(f)
    #         a = np.matmul(np.linalg.inv(np.matmul(cm.T,cm)),cm.T)
    #         drdx = np.matmul(a,fm)
    #         current_para = current_para + lr * drdx
    #         j = self.objective(current_para[0:3],current_para[3:6],t)
    #         if((self.objective(current_para[0:3],current_para[3:6],t+0.1)-j)>20):
    #             t = t + 0.1
    #         else:
    #             if((self.objective(current_para[0:3],current_para[3:6],t-0.1)-j)>20):
    #                 t = t - 0.1
    #         t = round(t,1) 
    #         print(str(t)+str('  ')+str(drdx)+str('  ')+str(k))
    #     return [current_para, j,self.collision,self.path]        

    # ## play the animation for one set of high-level paramters of such a scenario
    # def play_ani(self, tra_pos=None,tra_ang=None, t = 3,Ulast = None):
    #     tra_atti = Rd2Rp(tra_ang)
    #     self.uav1.init_TraCost(tra_pos,tra_atti)
    #     self.uavoc1.setTraCost(self.uav1.tra_cost,t)
    #     ## obtain the trajectory
    #     self.sol1 = self.uavoc1.ocSolver(ini_state=self.ini_state, horizon=self.horizon,dt=self.dt,Ulast=Ulast)
    #     state_traj1 = self.sol1['state_traj_opt']
    #     traj = self.uav1.get_quadrotor_position(wing_len = self.winglen, state_traj = state_traj1)
    #     ## plot the animation
    #     self.uav1.play_animation(wing_len = self.winglen, state_traj = state_traj1,dt=self.dt, point1 = self.point1,\
    #         point2 = self.point2, point3 = self.point3, point4 = self.point4)
    
    # ## given initial state, control command, high-level parameters, obtain the first control command of the quadrotor
    # def get_input(self, ini_state, Ulast ,tra_pos, tra_ang, t):
    #     tra_atti = Rd2Rp(tra_ang)
    #     # initialize the NLP problem
    #     self.uav1.init_TraCost(tra_pos,tra_atti)
    #     self.uavoc1.setTraCost(self.uav1.tra_cost,t)
    #     ## obtain the solution
    #     self.sol1 = self.uavoc1.ocSolver(ini_state=ini_state,horizon=self.horizon,dt=self.dt, Ulast=Ulast)
    #     # obtain the control command
    #     control = self.sol1['control_traj_opt'][0,:]
    #     return control

## sample the perturbation (only for random perturbations)
def sample(deviation):
    act = np.random.normal(0,deviation,size=6)
    return act
## this file is a package for policy search for quadrotor

from quad_OC import OCSys
from math import cos, pi, sin, sqrt, tan
from quad_model import *
from casadi import *
import scipy.io as sio
import numpy as np
from solid_geometry import *
def Rd2Rp(tra_ang):
    theta = 2*math.atan(magni(tra_ang))
    vector = norm(tra_ang+np.array([1e-8,0,0]))
    return [theta,vector]

class run_quad:
    def __init__(self, goal_pos = [0, 8, 0], goal_atti = [0,[1,0,0]], ini_r=[0,-8,0]\
            ,ini_v_I = [0.0, 0.0, 0.0], ini_q = toQuaternion(0.0,[3,3,5]),horizon = 50, 
            reward_weight = 1000, ep_beta = 0.01):
        ## definition 
        self.winglen = 1.5
        # goal
        self.goal_pos = goal_pos
        self.goal_atti = goal_atti 
        # initial
        if type(ini_r) is not list:
            ini_r = ini_r.tolist()
        self.ini_r = ini_r
        self.ini_v_I = ini_v_I 
        self.ini_q = ini_q
        self.ini_w =  [0.0, 0.0, 0.0]
        self.ini_state = self.ini_r + self.ini_v_I + self.ini_q + self.ini_w
        # set horizon
        self.horizon = horizon
        self.reward_weight = reward_weight
        self.beta = ep_beta
        print(f"ep_beta is {ep_beta}")

        # --------------------------- create model1 ----------------------------------------
        self.uav1 = Quadrotor()
        jx, jy, jz = 0.0023, 0.0023, 0.004
        self.uav1.initDyn(Jx=0.0023,Jy=0.0023,Jz=0.004,mass=0.5,l=0.35,c=0.0245) # hb quadrotor
        self.uav1.initCost(wrt=5,wqt=80,wthrust=0.1,wrf=5,wvf=5,wqf=0,wwf=3, weptl = self.beta, goal_pos=self.goal_pos) # wthrust = 0.1
        self.uav1.init_TraCost()

        # --------------------------- create PDP object1 ----------------------------------------
        # create a pdp object
        self.dt = 0.1
        self.uavoc1 = OCSys()
        self.uavoc1.setAuxvarVariable()
        sc   = 1e20
        wc   = pi/2 #pi
        tw   = 1.22
        t2w  = 2
        self.uavoc1.setStateVariable(self.uav1.X,state_lb=[-sc,-sc,-sc,-sc,-sc,-sc,-sc,-sc,-sc,-sc,-wc,-wc,-wc],state_ub=[sc,sc,sc,sc,sc,sc,sc,sc,sc,sc,wc,wc,wc])
        self.uavoc1.setControlVariable(self.uav1.U,control_lb=[0,0,0,0],control_ub= [t2w*tw,t2w*tw,t2w*tw,t2w*tw]) # thrust-to-weight = 4:1
        self.uavoc1.setDyn(self.uav1.f,self.dt)
        self.uavoc1.setthrustcost(self.uav1.thrust_cost)
        self.uavoc1.setPathCost(self.uav1.goal_cost)
        self.uavoc1.setTraCost(self.uav1.tra_cost)
        self.uavoc1.setFinalCost(self.uav1.final_cost)

    # define function
    # initialize the narrow window
    def init_obstacle(self,gate_point, alpha, beta, gamma):
        self.point1 = gate_point[0:3]
        self.point2 = gate_point[3:6]
        self.point3 = gate_point[6:9]
        self.point4 = gate_point[9:12]        
        # self.obstacle1 = obstacle(self.point1,self.point2,self.point3,self.point4)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.obstacle1 = obstacleNewReward(self.point1,self.point2,self.point3,self.point4, self.winglen/2, alpha, beta, gamma)

    def setObstacle(self):
        self.uav1.setObstacle(self.obstacle1, self.point1, self.point2, self.point3, self.point4, self.gamma, self.alpha, self.beta, self.reward_weight, self.winglen/2)
    
    def objective( self,ini_state = None,tra_pos=None,tra_ang=None,t = 3, Ulast = None, incl_ep = 0):
        if ini_state is None:
            ini_state = self.ini_state
        t = round(t,1)
        tra_atti = Rd2Rp(tra_ang)
        ## transfer the high-level parameters to traversal cost
        # define traverse cost
        self.uav1.init_TraCost(tra_pos,tra_atti)
        self.uavoc1.setTraCost(self.uav1.tra_cost, t)
        if incl_ep == 1:
            #include the ep loss in the loss function
            self.uav1.init_eptrajloss()
            self.uavoc1.setEpCost(self.uav1.ep_cost, t)
            self.uav1.init_eppathloss(self.reward_weight)
            self.uavoc1.setEpPathCost(self.uav1.ep_path_cost)
        # obtain solution of trajectory
        sol1 = self.uavoc1.ocSolver(ini_state=ini_state ,horizon=self.horizon,dt=self.dt, Ulast=Ulast, incl_ep = incl_ep)
        state_traj1 = sol1['state_traj_opt']
        self.traj = self.uav1.get_quadrotor_position(wing_len = self.winglen, state_traj = state_traj1)
        self.traj_quaternion = sol1['state_traj_opt'][:, 6:10]
        self.traj_pos = sol1['state_traj_opt'][:, :3]
        # calculate trajectory reward
        self.collision = 0
        self.path = 0
        ## detect whether there is detection
        self.co = 0
        curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4 = self.obstacle1.collis_det(self.traj_pos, self.horizon, self.traj_quaternion, t)
        self.collision += curr_collision
        self.co += self.obstacle1.co 
        for p in range(4):
            self.path += np.dot(self.traj[self.horizon-1-p,0:3]-self.goal_pos, self.traj[self.horizon-1-p,0:3]-self.goal_pos)
        reward = 1 * self.collision - (self.reward_weight * self.path)
        return reward, self.collision, curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4, self.path
    # --------------------------- solution and learning----------------------------------------
    ##solution and demo
    def sol_gradient(self,ini_state = None,tra_pos =None,tra_ang=None,t=None,Ulast=None):
        tra_ang = np.array(tra_ang)
        tra_pos = np.array(tra_pos)
        j, collision_full, curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4, path = self.objective (ini_state,tra_pos,tra_ang,t)
        traj_pos_free = self.traj_pos
        traj_quat_free = self.traj_quaternion
        trav_idx = t / 0.1
        pre_idx = int(np.floor(trav_idx))
        rp_free, ra_free = self.grad_computation(traj_pos_free, traj_quat_free, pre_idx, tra_pos,tra_ang,t)

        j_fix, collision_full_fix, curr_collision_fix, comp1_fix, comp2_fix, collide_fix, curr_delta1_fix, curr_delta2_fix, curr_delta3_fix, curr_delta4_fix, path_fix = self.objective (ini_state,tra_pos,tra_ang,t, incl_ep = 1)
        traj_pos_perturbed = self.traj_pos
        traj_quat_perturbed = self.traj_quaternion
        rp_perturbed, ra_perturbed = self.grad_computation(traj_pos_perturbed, traj_quat_perturbed, pre_idx, tra_pos,tra_ang,t)
        ## fixed perturbation to calculate the gradient
        delta = 1e-3
        # drdx = np.clip(self.objective(ini_state,tra_pos+[delta,0,0],tra_ang, t,Ulast)[0] - j,-0.5,0.5)*0.1
        # drdy = np.clip(self.objective(ini_state,tra_pos+[0,delta,0],tra_ang, t,Ulast)[0] - j,-0.5,0.5)*0.1
        # drdz = np.clip(self.objective(ini_state,tra_pos+[0,0,delta],tra_ang, t,Ulast)[0] - j,-0.5,0.5)*0.1
        # drda = np.clip(self.objective(ini_state,tra_pos,tra_ang+[delta,0,0], t,Ulast)[0] - j,-0.5,0.5)*(1/(500*tra_ang[0]**2+5))
        # drdb = np.clip(self.objective(ini_state,tra_pos,tra_ang+[0,delta,0], t,Ulast)[0] - j,-0.5,0.5)*(1/(500*tra_ang[1]**2+5))
        # drdc = np.clip(self.objective(ini_state,tra_pos,tra_ang+[0,0,delta], t,Ulast)[0] - j,-0.5,0.5)*(1/(500*tra_ang[2]**2+5))
        drdx = (rp_perturbed[0] - rp_free[0]) / self.beta
        drdy = (rp_perturbed[1] - rp_free[1]) / self.beta
        drdz = (rp_perturbed[2] - rp_free[2]) / self.beta
        drda = (ra_perturbed[0] - ra_free[0]) / self.beta
        drdb = (ra_perturbed[1] - ra_free[1]) / self.beta
        drdc = (ra_perturbed[2] - ra_free[2]) / self.beta

        drdt =0
        if((self.objective(ini_state,tra_pos,tra_ang,t-0.1)[0]-j)>2):
            drdt = -0.05
        if((self.objective(ini_state,tra_pos,tra_ang,t+0.1)[0]-j)>2):
            drdt = 0.05
        ## return gradient and reward (for deep learning)
        return np.array([-drdx,-drdy,-drdz,-drda,-drdb,-drdc,-drdt,j, collision_full, curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4, path])
    
    def grad_computation(self, traj, traj_q, idx, tra_pos,tra_ang,t):
        real_tra_pos = traj[idx]
        real_tra_q = traj_q[idx]


        self.r_I_gra = SX.sym('rxg', 3,1)

        # q0_g, q1_g, q2_g, q3_g = SX.sym('q0g'), SX.sym('q1g'), SX.sym('q2g'), SX.sym('q3g')
        # self.q_gra = vertcat(q0_g, q1_g, q2_g, q3_g)
        self.q_gra = SX.sym('q_g', 4, 1)

        # t_rx_g, t_ry_g, t_rz_g = SX.sym('trxg'), SX.sym('tryg'), SX.sym('trzg')
        # self.t_r_I_gra = vertcat(t_rx_g, t_ry_g, t_rz_g)
        self.t_r_I_gra = SX.sym('trxg', 3,1)

        # t_a_g, t_b_g, t_c_g = SX.sym('tag'), SX.sym('tbg'), SX.sym('tcg')
        # self.t_ang_gra = vertcat(t_a_g, t_b_g, t_c_g)
        self.t_ang_gra = SX.sym('tag', 3, 1)
        self.uav1.total_GraCost(self.r_I_gra,self.t_r_I_gra, self.t_ang_gra, self.q_gra )
        drp = casadi.jacobian(self.uav1.tra_cost_g, self.t_r_I_gra)
        dra = casadi.jacobian(self.uav1.tra_cost_g, self.t_ang_gra)

        f_p = casadi.Function('gra_p',[self.r_I_gra, self.q_gra, self.t_r_I_gra, self.t_ang_gra],[drp],['rxg','q_g', 'trxg', 'tag'],['rw'])
        f_a = casadi.Function('gra_p',[self.r_I_gra, self.q_gra, self.t_r_I_gra, self.t_ang_gra],[dra],['rxg','q_g', 'trxg', 'tag'],['rw'])

        real_tra_pos_casadi = casadi.SX([real_tra_pos[0],real_tra_pos[1],real_tra_pos[2]])
        real_tra_q_casadi = casadi.SX([real_tra_q[0],real_tra_q[1],real_tra_q[2], real_tra_q[3]])
        nn_tra_pos_casadi = casadi.SX([tra_pos[0],tra_pos[1],tra_pos[2]])
        nn_tra_q_casadi = casadi.SX([tra_ang[0], tra_ang[1], tra_ang[2]])
        rp = f_p(real_tra_pos_casadi, real_tra_q_casadi, nn_tra_pos_casadi, nn_tra_q_casadi)
        ra = f_p(real_tra_pos_casadi, real_tra_q_casadi, nn_tra_pos_casadi, nn_tra_q_casadi)

        return rp, ra




    def sol_test(self,ini_state = None,tra_pos =None,tra_ang=None,t=None,Ulast=None):
        tra_ang = np.array(tra_ang)
        tra_pos = np.array(tra_pos)
        j, collision_full, curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4, path = self.objective (ini_state,tra_pos,tra_ang,t)

        traj = self.traj_pos

        return np.array([j, collision_full, curr_collision, comp1, comp2, collide, curr_delta1, curr_delta2, curr_delta3, curr_delta4, path]), traj
    
    def optimize(self, t):
        tra_pos = self.obstacle1.centroid
        tra_posx = self.obstacle1.centroid[0]
        tra_posy = self.obstacle1.centroid[1]
        tra_posz = self.obstacle1.centroid[2]
        tra_a = 0
        tra_b = 0
        tra_c = 0
        tra_ang = np.array([tra_a,tra_b,tra_c])
        ## fixed perturbation to calculate the gradient
        for k in range(200):
            j = self.objective (tra_pos,tra_ang,t)
            drdx = np.clip(self.objective(tra_pos+[0.001,0,0],tra_ang=tra_ang, t=t) - j,-0.5,0.5)
            drdy = np.clip(self.objective(tra_pos+[0,0.001,0],tra_ang=tra_ang, t=t) - j,-0.5,0.5)
            drdz = np.clip(self.objective(tra_pos+[0,0,0.001],tra_ang=tra_ang, t=t) - j,-0.5,0.5)
            drda = np.clip(self.objective(tra_pos,tra_ang=tra_ang+[0.001,0,0], t=t) - j,-0.5,0.5)
            drdb = np.clip(self.objective(tra_pos,tra_ang=tra_ang+[0,0.001,0], t=t) - j,-0.5,0.5)
            drdc = np.clip(self.objective(tra_pos,tra_ang=tra_ang+[0,0,0.001], t=t) - j,-0.5,0.5)
            #drdt = np.clip(self.objective(tra_pos,tra_ang,t-0.1)-j,-10,10)
            # update
            tra_posx += 0.1*drdx
            tra_posy += 0.1*drdy
            tra_posz += 0.1*drdz
            tra_a += (1/(500*tra_a**2+5))*drda
            tra_b += (1/(500*tra_b**2+5))*drdb
            tra_c += (1/(500*tra_c**2+5))*drdc
            if((self.objective(tra_pos,tra_ang,t-0.1)-j)>2):
                t = t-0.1
            if((self.objective(tra_pos,tra_ang,t+0.1)-j)>2):
                t = t+0.1
            t = round(t,1)
            tra_pos = np.array([tra_posx,tra_posy,tra_posz])
            tra_ang = np.array([tra_a,tra_b,tra_c])
            ## display the process
            print(str(j)+str('  ')+str(tra_pos)+str('  ')+str(tra_ang)+str('  ')+str(t)+str('  ')+str(k))
        return [t,tra_posx,tra_posy,tra_posz,tra_a, tra_b,tra_c, j,self.collision,self.path]

    ## use random perturbations to calculate the gradient and update(not recommonded)
    def LSFD(self,t):
        tra_posx = self.obstacle1.centroid[0]
        tra_posy = self.obstacle1.centroid[1]
        tra_posz = self.obstacle1.centroid[2]
        tra_a = 0
        tra_b = 0
        tra_c = 0
        current_para = np.array([tra_posx,tra_posy,tra_posz,tra_a,tra_b,tra_c])
        lr = np.array([2e-4,2e-4,2e-4,5e-5,5e-5,5e-5])
        for k in range(50):
            j = self.objective(current_para[0:3],current_para[3:6],t)
            # calculate derivatives
            c = []
            f = []
            for i in range(24):
                dx = sample(0.001)
                dr = self.objective (current_para[0:3]+dx[0:3],current_para[3:6]+dx[3:6],t)-j
                c += [dx]
                f += [dr]
            # update
            cm = np.array(c)
            fm = np.array(f)
            a = np.matmul(np.linalg.inv(np.matmul(cm.T,cm)),cm.T)
            drdx = np.matmul(a,fm)
            current_para = current_para + lr * drdx
            j = self.objective(current_para[0:3],current_para[3:6],t)
            if((self.objective(current_para[0:3],current_para[3:6],t+0.1)-j)>20):
                t = t + 0.1
            else:
                if((self.objective(current_para[0:3],current_para[3:6],t-0.1)-j)>20):
                    t = t - 0.1
            t = round(t,1) 
            print(str(t)+str('  ')+str(drdx)+str('  ')+str(k))
        return [current_para, j,self.collision,self.path]        

    ## play the animation for one set of high-level paramters of such a scenario
    def play_ani(self, tra_pos=None,tra_ang=None, t = 3,Ulast = None):
        tra_atti = Rd2Rp(tra_ang)
        self.uav1.init_TraCost(tra_pos,tra_atti)
        self.uavoc1.setTraCost(self.uav1.tra_cost,t)
        ## obtain the trajectory
        self.sol1 = self.uavoc1.ocSolver(ini_state=self.ini_state, horizon=self.horizon,dt=self.dt,Ulast=Ulast)
        state_traj1 = self.sol1['state_traj_opt']
        traj = self.uav1.get_quadrotor_position(wing_len = self.winglen, state_traj = state_traj1)
        ## plot the animation
        self.uav1.play_animation(wing_len = self.winglen, state_traj = state_traj1,dt=self.dt, point1 = self.point1,\
            point2 = self.point2, point3 = self.point3, point4 = self.point4)
    
    ## given initial state, control command, high-level parameters, obtain the first control command of the quadrotor
    def get_input(self, ini_state, Ulast ,tra_pos, tra_ang, t):
        tra_atti = Rd2Rp(tra_ang)
        # initialize the NLP problem
        self.uav1.init_TraCost(tra_pos,tra_atti)
        self.uavoc1.setTraCost(self.uav1.tra_cost,t)
        ## obtain the solution
        self.sol1 = self.uavoc1.ocSolver(ini_state=ini_state,horizon=self.horizon,dt=self.dt, Ulast=Ulast)
        # obtain the control command
        control = self.sol1['control_traj_opt'][0,:]
        return control

## sample the perturbation (only for random perturbations)
def sample(deviation):
    act = np.random.normal(0,deviation,size=6)
    return act
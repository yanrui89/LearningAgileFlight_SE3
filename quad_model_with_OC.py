##this file is to generate model of quadrotor

from casadi import *
import casadi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from scipy.spatial.transform import Rotation as R
from solid_geometry import norm
from math import sqrt
from solid_geometry import magni

# quadrotor (UAV) environment
class QuadrotorOC:
    def __init__(self, project_name='my UAV'):
        self.project_name = 'my uav'

        # define the state of the quadrotor
        rx, ry, rz = SX.sym('rx'), SX.sym('ry'), SX.sym('rz')
        self.r_I = vertcat(rx, ry, rz)
        vx, vy, vz = SX.sym('vx'), SX.sym('vy'), SX.sym('vz')
        self.v_I = vertcat(vx, vy, vz)
        # quaternions attitude of B w.r.t. I
        q0, q1, q2, q3 = SX.sym('q0'), SX.sym('q1'), SX.sym('q2'), SX.sym('q3')
        self.q = vertcat(q0, q1, q2, q3)
        wx, wy, wz = SX.sym('wx'), SX.sym('wy'), SX.sym('wz')
        self.w_B = vertcat(wx, wy, wz)
        # define the quadrotor input
        f1, f2, f3, f4 = SX.sym('f1'), SX.sym('f2'), SX.sym('f3'), SX.sym('f4')
        self.T_B = vertcat(f1, f2, f3, f4)
        # define total thrust and control torques
        # self.thrust, self.Mx, self.My, self.Mz = SX.sym('T'), SX.sym('Mx'), SX.sym('My'), SX.sym('Mz')
        # self.U   = vertcat(self.thrust, self.Mx, self.My, self.Mz)

    def initDyn(self, Jx=None, Jy=None, Jz=None, mass=None, l=None, c=None):
        # global parameter
        g = 9.78
        # parameters settings

        self.Jx = Jx

        self.Jy = Jy

        self.Jz = Jz

        self.mass = mass

        self.l = l

        self.c = c

        # Angular moment of inertia
        self.J_B = diag(vertcat(self.Jx, self.Jy, self.Jz))
        # Gravity
        self.g_I = vertcat(0, 0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        # total thrust in body frame
        thrust = self.T_B[0] + self.T_B[1] + self.T_B[2] + self.T_B[3]
        self.thrust_B = vertcat(0, 0, thrust)
        # total moment M in body frame
        Mx = -self.T_B[1] * self.l / 2 + self.T_B[3] * self.l / 2
        My = -self.T_B[0] * self.l / 2 + self.T_B[2] * self.l / 2
        Mz = (self.T_B[0] - self.T_B[1] + self.T_B[2] - self.T_B[3]) * self.c

        self.u_m = np.array([
            [1,1,1,1],
            [0,-self.l/2,0,self.l/2],
            [-self.l/2,0,self.l/2,0],
            [self.c,-self.c,self.c,-self.c]
        ])
        self.M_B = vertcat(Mx, My, Mz)

        # cosine directional matrix
        C_B_I = self.dir_cosine(self.q)  # inertial to body
        C_I_B = transpose(C_B_I)  # body to inertial

        # Newton's law
        dr_I = self.v_I
        dv_I = 1 / self.m * mtimes(C_I_B, self.thrust_B) + self.g_I
        # Euler's law
        dq = 1 / 2 * mtimes(self.omega(self.w_B), self.q)
        dw = mtimes(inv(self.J_B), self.M_B - mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

        self.X = vertcat(self.r_I, self.v_I, self.q, self.w_B)
        self.U = self.T_B
        self.f = vertcat(dr_I, dv_I, dq, dw)

    def initCost(self, wrt=None, wqt=None, wrf=None, wvf=None, wqf=None, wwf=None, weptl = None, \
        wthrust=0.5,goal_pos=[0,9,5],goal_velo = [0,0,0],goal_atti=[0,[1,0,0]]):
        #traverse

        # traversal position weight
        self.wrt = wrt

        # traversal attitude weight
        self.wqt = wqt

        #final position weight
        self.wrf = wrf
        
        #final velocity weight
        self.wvf = wvf
        
        #final attitude weight
        self.wqf = wqf
        
        #final omega weight
        self.wwf = wwf

        #ep_beta
        self.weptl = weptl

        #thrust cost
        self.wthrust = wthrust

        ## ----------------------goal cost-------------------------------#
        # goal position in the world frame
        self.goal_r_I = goal_pos
        self.cost_r_I_g = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)

        # goal velocity
        self.goal_v_I = goal_velo
        self.cost_v_I_g = dot(self.v_I - self.goal_v_I, self.v_I - self.goal_v_I)

        # final attitude error
        self.goal_q = toQuaternion(goal_atti[0],goal_atti[1])
        goal_R_B_I = self.dir_cosine(self.goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q_g = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # auglar velocity cost
        self.goal_w_B = [0, 0, 0]
        self.cost_w_B_g = dot(self.w_B - self.goal_w_B, self.w_B - self.goal_w_B)

        ## the final (goal) cost
        self.goal_cost = self.wrf * self.cost_r_I_g + \
                         self.wvf * self.cost_v_I_g + \
                         self.wwf * self.cost_w_B_g + \
                         self.wqf * self.cost_q_g 
        self.setPathCost()
        
        
        self.final_cost = self.wrf * self.cost_r_I_g + \
                          self.wvf * self.cost_v_I_g + \
                          self.wwf * self.cost_w_B_g + \
                          self.wqf * self.cost_q_g
        
        self.setFinalCost()

        # -------------------------the thrust cost ----------------------------#
        self.cost_torque = dot(self.T_B, self.T_B)
        self.thrust_cost = self.wthrust * (self.cost_torque)
        self.setthrustcost() 

    def init_TraCost(self, tra_pos = [0, 0, 5], tra_atti = [0.7,[0,1,0]], tra_t = 1.0): # transforming Rodrigues to Quaternion is shown in get_input function
        ## traverse cost
        # traverse position in the world frame
        self.tra_r_I = tra_pos[0:3]
        self.cost_r_I_t = dot(self.r_I - self.tra_r_I, self.r_I - self.tra_r_I)

        # traverse attitude error
        self.tra_q = toQuaternion(tra_atti[0],tra_atti[1])
        tra_R_B_I = self.dir_cosine(self.tra_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q_t = trace(np.identity(3) - mtimes(transpose(tra_R_B_I), R_B_I))**2

        self.tra_cost =   self.wrt * self.cost_r_I_t + \
                            self.wqt * self.cost_q_t
        
        self.t = tra_t
        self.setTraCost()

    def init_eptrajloss(self):
        self.eptrajcost = self.collis_det_ep2(self.r_I, self.q, False)
        self.ep_cost = self.weptl * self.eptrajcost
        self.setEpCost()

    def init_eppathloss(self, reward_weight):
        self.goalcost_r_I_t = dot(self.r_I - self.goal_r_I, self.r_I - self.goal_r_I)
        self.ep_path_cost = self.weptl * self.goalcost_r_I_t * reward_weight
        self.setEpPathCost()

    def setObstacle(self,point1, point2, point3, point4, gamma, alpha, beta, reward_weight, wingrad):
        # For reward computation
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        self.point4 = point4
        self.points = [point1, point2, point3, point4]
        self.midpoint1 = (self.point1 + self.point2)/2
        self.midpoint2 = (self.point2 + self.point3)/2
        self.midpoint3 = (self.point3 + self.point4)/2
        self.midpoint4 = (self.point4 + self.point1)/2
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reward_weight = reward_weight
        self.wingrad = wingrad
        self.height = 0.3
        self.R_max = 100.0
        plane_ref_vect1 = point2 - point1
        plane_ref_vect2 = point3 - point1
        
        self.plane_norm = np.cross(plane_ref_vect1, plane_ref_vect2)
        self.area = casadi.sqrt(casadi.dot((self.point2 - self.point1), (self.point2 - self.point1)) * casadi.dot((self.point3 - self.point2), (self.point3 - self.point2)))
        self.plane_norm_unit = self.plane_norm / np.linalg.norm(self.plane_norm)

    def collis_det(self, vert_traj, quat, trav_t, verbose):
        ## define the state whether find corresponding plane
        trav_idx = trav_t / 0.1
        pre_idx = int(np.floor(trav_idx))
        # post_idx = np.floor(trav_idx)
        # intersect = self.plane1.interpoint(vert_traj[t],vert_traj[t-1])
        X = vert_traj[pre_idx]
        # judge whether they belong to plane1 and calculate the distance
        curr_quat = quat[pre_idx,:]

        reward_data = self.collis_det_ep2(X, curr_quat, verbose)

        if verbose == True:
            col_reward, curr_col1, curr_col2,co, curr_delta1, curr_delta2, curr_delta3, curr_delta4 = reward_data
            return col_reward, curr_col1, curr_col2,co, curr_delta1, curr_delta2, curr_delta3, curr_delta4
        else:
            col_reward = reward_data
            return col_reward

    def collis_det_ep(self, X, curr_quat, verbose):

        self.scaledMatrix = casadi.SX([[self.wingrad,0,0],[0,self.wingrad, 0],[0,0,self.height]])
        # self.scaledMatrix = np.diag([self.wingrad, self.wingrad, self.height])  
        curr_r_I2B = self.dir_cosine(curr_quat)
        # curr_r_I2B = self.quaternion_rotation_matrix(self.q)
        # curr_r_I2B_chk = self.quaternion_rotation_matrix(curr_quat)
        E = mtimes(self.scaledMatrix, transpose(curr_r_I2B))
        E = mtimes(curr_r_I2B, E)

        # curr_delta11 = mtimes(casadi.inv(E), (self.midpoint1 - X))
        # curr_delta1 = casadi.sqrt(curr_delta11[0]*curr_delta11[0] + curr_delta11[1]*curr_delta11[1] + curr_delta11[2]*curr_delta11[2])
        # curr_delta22 = mtimes(casadi.inv(E), (self.midpoint2 - X))
        # curr_delta2 = casadi.sqrt(curr_delta22[0]*curr_delta22[0] + curr_delta22[1]*curr_delta22[1] + curr_delta22[2]*curr_delta22[2])
        # curr_delta33 = mtimes(casadi.inv(E), (self.midpoint3 - X))
        # curr_delta3 = casadi.sqrt(curr_delta33[0]*curr_delta33[0] + curr_delta33[1]*curr_delta33[1] + curr_delta33[2]*curr_delta33[2])
        # curr_delta44 = mtimes(casadi.inv(E), (self.midpoint4 - X))
        # curr_delta4 = casadi.sqrt(curr_delta44[0]*curr_delta44[0] + curr_delta44[1]*curr_delta44[1] + curr_delta44[2]*curr_delta44[2])
        curr_col1 = 0
        curr_col2 = 0
        co = 0

        for i in range(4):
            idx_start = i
            idx_end = (i+1)% 4
            point_start = self.points[idx_start]
            point_end = self.points[idx_end]
            diff_dist = (point_end - point_start)
            diff_norm = casadi.sqrt(diff_dist[0]*diff_dist[0] + diff_dist[1]*diff_dist[1] + diff_dist[2]*diff_dist[2])
            unit_diff = diff_dist/diff_norm
            interval_diff_norm = diff_norm / 10
            for j in range(10):
                curr_pt = interval_diff_norm*j*unit_diff + point_start
                curr_delta_int = mtimes(casadi.inv(E), (curr_pt - X))
                curr_delta = casadi.sqrt(curr_delta_int[0]*curr_delta_int[0] + curr_delta_int[1]*curr_delta_int[1] + curr_delta_int[2]*curr_delta_int[2])
                curr_col1 += self.gamma  * casadi.power(curr_delta, 2)
                curr_col2 += self.alpha * casadi.exp(self.beta*(1 - curr_delta))
                if verbose == True:
                    if curr_delta < 1:
                        co += 0

            if verbose == True:
                curr_delta1 = curr_delta
                curr_delta2 = curr_delta
                curr_delta3 = curr_delta
                curr_delta4 = curr_delta
                # if curr_delta < 1:
                #     co += 1
                    
                # if curr_delta2 < 1:
                #     co += 1
                # if curr_delta3 < 1:
                #     co+=1
                # if curr_delta4 < 1:
                #     co += 1


        # curr_col1 = (self.gamma  * casadi.power(curr_delta1, 2)) + (self.gamma  * casadi.power(curr_delta2, 2)) + (self.gamma  * casadi.power(curr_delta3, 2)) + (self.gamma  * casadi.power(curr_delta4, 2)) #+ (self.gamma  * np.power(curr_delta11, 2)) + (self.gamma  * np.power(curr_delta22, 2)) + (self.gamma  * np.power(curr_delta33, 2)) + (self.gamma  * np.power(curr_delta44, 2))
        # curr_col2 = (self.alpha * casadi.exp(self.beta*(1 - curr_delta1))) + (self.alpha * casadi.exp(self.beta*(1 - curr_delta2))) + (self.alpha * casadi.exp(self.beta*(1 - curr_delta3))) + (self.alpha * casadi.exp(self.beta*(1 - curr_delta4))) #+ (self.alpha * np.exp(self.beta*(1 - curr_delta11))) + (self.alpha * np.exp(self.beta*(1 - curr_delta22))) + (self.alpha * np.exp(self.beta*(1 - curr_delta33))) + (self.alpha * np.exp(self.beta*(1 - curr_delta44)))

        collision = curr_col1 + curr_col2

        if verbose == True:
            return collision, curr_col1, curr_col2, co, curr_delta1, curr_delta2, curr_delta3, curr_delta4
        else:
            return collision
        
    def collis_det_ep2(self, X, curr_quat, verbose):
        
        #project the position to the plane and check 2 things. How far it is and if it is within the polygon
        rel_dist = X - self.point1
        rel_dist_norm = casadi.dot(rel_dist, self.plane_norm_unit)
        proj_point = (rel_dist - (rel_dist_norm * self.plane_norm_unit)) + self.point1

        total_area = 0
        for i in range(4):
            j = (i+1)% 4
            area = self.area_of_triangle(self.points[i][0], self.points[i][2], self.points[j][0], self.points[j][2], proj_point[0], proj_point[2])
            total_area += area

        area_ratio = total_area / self.area   
        
        
        
    
        self.scaledMatrix = casadi.SX([[self.wingrad,0,0],[0,self.wingrad, 0],[0,0,self.height]])
        # self.scaledMatrix = np.diag([self.wingrad, self.wingrad, self.height])  
        curr_r_I2B = self.dir_cosine(curr_quat)
        # curr_r_I2B = self.quaternion_rotation_matrix(self.q)
        # curr_r_I2B_chk = self.quaternion_rotation_matrix(curr_quat)
        E = mtimes(self.scaledMatrix, transpose(curr_r_I2B))
        E = mtimes(curr_r_I2B, E)

        # curr_delta11 = mtimes(casadi.inv(E), (self.midpoint1 - X))
        # curr_delta1 = casadi.sqrt(curr_delta11[0]*curr_delta11[0] + curr_delta11[1]*curr_delta11[1] + curr_delta11[2]*curr_delta11[2])
        # curr_delta22 = mtimes(casadi.inv(E), (self.midpoint2 - X))
        # curr_delta2 = casadi.sqrt(curr_delta22[0]*curr_delta22[0] + curr_delta22[1]*curr_delta22[1] + curr_delta22[2]*curr_delta22[2])
        # curr_delta33 = mtimes(casadi.inv(E), (self.midpoint3 - X))
        # curr_delta3 = casadi.sqrt(curr_delta33[0]*curr_delta33[0] + curr_delta33[1]*curr_delta33[1] + curr_delta33[2]*curr_delta33[2])
        # curr_delta44 = mtimes(casadi.inv(E), (self.midpoint4 - X))
        # curr_delta4 = casadi.sqrt(curr_delta44[0]*curr_delta44[0] + curr_delta44[1]*curr_delta44[1] + curr_delta44[2]*curr_delta44[2])
        curr_col1 = 0
        curr_col2 = 0
        co = 0

        for i in range(4):
            idx_start = i
            idx_end = (i+1)% 4
            point_start = self.points[idx_start]
            point_end = self.points[idx_end]
            diff_dist = (point_end - point_start)
            diff_norm = casadi.sqrt(diff_dist[0]*diff_dist[0] + diff_dist[1]*diff_dist[1] + diff_dist[2]*diff_dist[2])
            unit_diff = diff_dist/diff_norm
            interval_diff_norm = diff_norm / 10
            for j in range(10):
                curr_pt = interval_diff_norm*j*unit_diff + point_start
                curr_delta_int = mtimes(casadi.inv(E), (curr_pt - X))
                curr_delta = casadi.sqrt(curr_delta_int[0]*curr_delta_int[0] + curr_delta_int[1]*curr_delta_int[1] + curr_delta_int[2]*curr_delta_int[2])
                # curr_col1 += self.alpha  * casadi.exp(self.beta*(area_ratio)) + self.gamma * casadi.power(rel_dist_norm, 2)
                curr_col2 += self.alpha * casadi.exp(self.beta*(1 - curr_delta))
                if verbose == True:
                    if curr_delta < 1:
                        co += 0
        
            if verbose == True:
                curr_delta1 = curr_delta
                curr_delta2 = curr_delta
                curr_delta3 = curr_delta
                curr_delta4 = curr_delta
                # if curr_delta < 1:
                #     co += 1
                    
                # if curr_delta2 < 1:
                #     co += 1
                # if curr_delta3 < 1:
                #     co+=1
                # if curr_delta4 < 1:
                #     co += 1

        curr_col1 += self.alpha  * casadi.exp(self.beta*(area_ratio - 1)) + self.gamma * casadi.power(rel_dist_norm, 2)
        # curr_col1 = (self.gamma  * casadi.power(curr_delta1, 2)) + (self.gamma  * casadi.power(curr_delta2, 2)) + (self.gamma  * casadi.power(curr_delta3, 2)) + (self.gamma  * casadi.power(curr_delta4, 2)) #+ (self.gamma  * np.power(curr_delta11, 2)) + (self.gamma  * np.power(curr_delta22, 2)) + (self.gamma  * np.power(curr_delta33, 2)) + (self.gamma  * np.power(curr_delta44, 2))
        # curr_col2 = (self.alpha * casadi.exp(self.beta*(1 - curr_delta1))) + (self.alpha * casadi.exp(self.beta*(1 - curr_delta2))) + (self.alpha * casadi.exp(self.beta*(1 - curr_delta3))) + (self.alpha * casadi.exp(self.beta*(1 - curr_delta4))) #+ (self.alpha * np.exp(self.beta*(1 - curr_delta11))) + (self.alpha * np.exp(self.beta*(1 - curr_delta22))) + (self.alpha * np.exp(self.beta*(1 - curr_delta33))) + (self.alpha * np.exp(self.beta*(1 - curr_delta44)))

        collision = curr_col1 + curr_col2

        if verbose == True:
            return collision, curr_col1, curr_col2, co, curr_delta1, curr_delta2, curr_delta3, curr_delta4
        else:
            return collision
        
    def area_of_triangle(self, x1, y1, x2, y2, x3, y3):
        temp_a = 0.5 * (x1 *(y2-y3) + x2* (y3 - y1) + x3 * (y1 - y2))
        temp_b = temp_a * temp_a
        return casadi.sqrt(temp_b)
        
    def collis_det_ep_trial(self,curr_quat):

        self.scaledMatrix = casadi.SX([[self.wingrad,0,0],[0,self.wingrad, 0],[0,0,self.height]])
        # self.scaledMatrix = np.diag([self.wingrad, self.wingrad, self.height])  
        curr_r_I2B = self.dir_cosine(curr_quat)
        # curr_r_I2B = self.quaternion_rotation_matrix(self.q)
        # curr_r_I2B_chk = self.quaternion_rotation_matrix(curr_quat)
        E = mtimes(self.scaledMatrix, transpose(curr_r_I2B))
        E = mtimes(curr_r_I2B, E)

        total_list = []

        for i in np.arange(-0.75, 0.75, 0.05):
            for j in np.arange(-0.75, 0.75, 0.05):
                for k in np.arange(-0.75, 0.75, 0.05):
                    curr_delta11 = mtimes(casadi.inv(E), casadi.SX([i,j,k]))
                    curr_delta1 = casadi.sqrt(curr_delta11[0]*curr_delta11[0] + curr_delta11[1]*curr_delta11[1] + curr_delta11[2]*curr_delta11[2])
            
                    if curr_delta1 < 1 and curr_delta1 > 0.9:
                        total_list.append(np.array([i,j,k]))


        return total_list
        
    def setthrustcost(self):
        assert self.thrust_cost.numel() == 1, "thrust_cost must be a scalar function"        
        self.thrust_cost_fn = casadi.Function('thrust_cost',[self.control], [self.thrust_cost])

    def setPathCost(self):
        assert self.goal_cost.numel() == 1, "path_cost must be a scalar function"
        self.path_cost_fn = casadi.Function('path_cost', [self.state], [self.goal_cost])

    def setFinalCost(self):
        assert self.goal_cost.numel() == 1, "final_cost must be a scalar function"
        self.final_cost_fn = casadi.Function('final_cost', [self.state], [self.final_cost])

    def setTraCost(self):
        self.tra_cost_fn = casadi.Function('tra_cost', [self.state], [self.tra_cost])

    def setEpCost(self):
        self.ep_cost_fn = casadi.Function('ep_cost', [self.state], [self.ep_cost])

    def setEpPathCost(self):
        self.ep_path_cost_fn = casadi.Function('ep_cost', [self.state], [self.ep_path_cost])

    def setDyn(self, dt):       
        # self.dyn = casadi.Function('f',[self.X, self.U],[self.f])
        self.dyn = self.X + dt * self.f
        self.dyn_fn = casadi.Function('dynamics', [self.X, self.U], [self.dyn])

        
        
    def total_GraCost(self, r_I_gra, t_r_I_gra, t_ang_gra, q_gra, traversal_time, idx_t, time_step):
        # rx_g, ry_g, rz_g = SX.sym('rxg'), SX.sym('ryg'), SX.sym('rzg')
        # self.r_I_gra = vertcat(rx_g, ry_g, rz_g)
         ## traverse cost
        # traverse position in the world frame
        self.cost_r_I_t_g = dot(r_I_gra - t_r_I_gra, r_I_gra - t_r_I_gra)

        # traverse attitude error
        self.tra_ang_theta, self.tra_axis = self.Rd2Rp2(t_ang_gra)
        self.tra_q_g = self.toQuaternion_casa(self.tra_ang_theta,self.tra_axis)
        tra_R_B_I_q = self.dir_cosine(self.tra_q_g)
        R_B_I_q = self.dir_cosine(q_gra)
        self.cost_q_t_g = trace(np.identity(3) - mtimes(transpose(tra_R_B_I_q), R_B_I_q))**2

        self.tra_cost_g =   self.wrt * self.cost_r_I_t_g + \
                            self.wqt * self.cost_q_t_g
        
        weight = 60*casadi.exp(-10*(time_step*idx_t-traversal_time)**2) 

        self.tra_cost_g = self.tra_cost_g * weight

    def total_GraCost_time_grad(self, r_I_gra, q_gra, t_r_I_gra, t_ang_gra, traversal_time, idx_t, time_step):
        # rx_g, ry_g, rz_g = SX.sym('rxg'), SX.sym('ryg'), SX.sym('rzg')
        # self.r_I_gra = vertcat(rx_g, ry_g, rz_g)
         ## traverse cost
        # traverse position in the world frame
        self.cost_r_I_t_g = dot(r_I_gra - t_r_I_gra, r_I_gra - t_r_I_gra)

        # traverse attitude error
        self.tra_ang_theta, self.tra_axis = self.Rd2Rp2(t_ang_gra)
        self.tra_q_g = self.toQuaternion_casa(self.tra_ang_theta,self.tra_axis)
        tra_R_B_I_q = self.dir_cosine(self.tra_q_g)
        R_B_I_q = self.dir_cosine(q_gra)
        self.cost_q_t_g = trace(np.identity(3) - mtimes(transpose(tra_R_B_I_q), R_B_I_q))**2

        self.tra_cost_g =   self.wrt * self.cost_r_I_t_g + \
                            self.wqt * self.cost_q_t_g
        
        weight = 60*casadi.exp(-10*(time_step*idx_t-traversal_time)**2) 

        grad = self.tra_cost_g * weight * (20*(time_step*idx_t-traversal_time))

        return grad

    ## below is for animation (demo)
    def get_quadrotor_position(self, wing_len, state_traj):

        # thrust_position in body frame
        r1 = vertcat(wing_len*0.5/ sqrt(2) , wing_len*0.5/ sqrt(2) , 0)
        r2 = vertcat(-wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r3 = vertcat(-wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)
        r4 = vertcat(wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)

        # r1 = vertcat(wing_len*0.5, 0, 0)
        # r2 = vertcat(0,-wing_len*0.5, 0)
        # r3 = vertcat(-wing_len*0.5,0, 0)
        # r4 = vertcat(0, wing_len*0.5, 0)
        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 15))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:3]
            # altitude of quaternion
            q = state_traj[t, 6:10]

            # direction cosine matrix from body to inertial
            CIB = np.transpose(self.dir_cosine(q).full())

            # position of each rotor in inertial frame
            r1_pos = rc + mtimes(CIB, r1).full().flatten()
            r2_pos = rc + mtimes(CIB, r2).full().flatten()
            r3_pos = rc + mtimes(CIB, r3).full().flatten()
            r4_pos = rc + mtimes(CIB, r4).full().flatten()

            # store
            position[t, 0:3] = rc
            position[t, 3:6] = r1_pos
            position[t, 6:9] = r2_pos
            position[t, 9:12] = r3_pos
            position[t, 12:15] = r4_pos

        return position
    
    def get_final_position(self,wing_len, p= None,q = None):
        p = self.tra_r_I
        q = self.tra_q
        r1 = vertcat(wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r2 = vertcat(-wing_len*0.5 / sqrt(2), wing_len*0.5 / sqrt(2), 0)
        r3 = vertcat(-wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)
        r4 = vertcat(wing_len*0.5 / sqrt(2), -wing_len*0.5 / sqrt(2), 0)

        # r1 = vertcat(wing_len*0.5, 0, 0)
        # r2 = vertcat(0,-wing_len*0.5, 0)
        # r3 = vertcat(-wing_len*0.5,0, 0)
        # r4 = vertcat(0, wing_len*0.5, 0)

        CIB = np.transpose(self.dir_cosine(q).full())
 
        r1_pos = p + mtimes(CIB, r1).full().flatten()   
        r2_pos = p + mtimes(CIB, r2).full().flatten()
        r3_pos = p + mtimes(CIB, r3).full().flatten()
        r4_pos = p + mtimes(CIB, r4).full().flatten()

        position = np.zeros(15)
        position[0:3] = p
        position[3:6] = r1_pos
        position[6:9] = r2_pos
        position[9:12] = r3_pos
        position[12:15] = r4_pos

        return position
    

# ---------------------------------- OC Stuffs ------------------------------------#
    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        # Setting the lower and upper bound
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def ocSolver(self, ini_state, Ulast=None, horizon=None, auxvar_value=1, print_level=0, dt = 0.1,costate_option=0, incl_ep = 0):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'goal_cost'), "Define the running cost function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost function first!"

        if type(ini_state) == numpy.ndarray:
            ini_state = ini_state.flatten().tolist()

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += ini_state
        ubw += ini_state
        w0 += ini_state
        if Ulast is not None:
            Ulast = Ulast
        else:
            Ulast = np.array([0,0,0,0])

        trav_idx = self.t / 0.1
        pre_idx = int(np.floor(trav_idx))
        
        # Formulate the NLP
        for k in range(int(horizon)):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.control_lb
            ubw += self.control_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.control_lb, self.control_ub)]

            #calculate weight
            weight = 3000*casadi.exp(-10*(dt*k-self.t)**2) #gamma should increase as the flight duration decreases
             
            # Integrate till the end of the interval
            Xnext = self.dyn_fn(Xk, Uk)
            Ck = weight*self.tra_cost_fn(Xk) + self.path_cost_fn(Xk)\
                +self.thrust_cost_fn(Uk) + 1*dot(Uk-Ulast,Uk-Ulast)
            J = J + Ck

            
            if incl_ep ==1 and k == pre_idx:
                J = J + self.ep_cost_fn(Xk)

            if incl_ep == 1:
                if int(horizon) - k < 5 and int(horizon) - k > 0: 
                    J = J + self.ep_path_cost_fn(Xk)

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.state_lb
            ubw += self.state_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.state_lb, self.state_ub)]
            Ulast = Uk

            # Add equality constraint
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Adding the final cost
        J = J + self.final_cost_fn(Xk)

        # Create an NLP solver and solve it
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # take the optimal control and state
        sol_traj = numpy.concatenate((w_opt, self.n_control * [0]))
        sol_traj = numpy.reshape(sol_traj, (-1, self.n_state + self.n_control))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        control_traj_opt = numpy.delete(sol_traj[:, self.n_state:], -1, 0)
        time = numpy.array([k for k in range(horizon + 1)])

        # Compute the costates using two options
        if costate_option == 0:
            # Default option, which directly obtains the costates from the NLP solver
            costate_traj_opt = numpy.reshape(sol['lam_g'].full().flatten(), (-1, self.n_state))
        else:
            # Another option, which solve the costates by the Pontryagin's Maximum Principle
            # The variable name is consistent with the notations used in the PDP paper
            dfx_fun = casadi.Function('dfx', [self.state, self.control], [jacobian(self.dyn, self.state)])
            dhx_fun = casadi.Function('dhx', [self.state], [jacobian(self.goal_cost, self.state)])
            dcx_fun = casadi.Function('dcx', [self.state, self.control],
                                      [jacobian(self.goal_cost, self.state)])
            costate_traj_opt = numpy.zeros((horizon, self.n_state))
            costate_traj_opt[-1, :] = dhx_fun(state_traj_opt[-1, :], auxvar_value)
            for k in range(horizon - 1, 0, -1):
                costate_traj_opt[k - 1, :] = dcx_fun(state_traj_opt[k, :], control_traj_opt[k, :],
                                                     auxvar_value).full() + numpy.dot(
                    numpy.transpose(dfx_fun(state_traj_opt[k, :], control_traj_opt[k, :], auxvar_value).full()),
                    costate_traj_opt[k, :])

        # output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "control_traj_opt": control_traj_opt,
                   "costate_traj_opt": costate_traj_opt,
                   "time": time,
                   "horizon": horizon,
                   "cost": sol['f'].full()}

        return opt_sol
    
# ----------------------------------- Utility Functions ---------------------------- #
    def Rd2Rp2(self,tra_ang):
        theta = 2*casadi.atan(self.magnit(tra_ang))
        vector = self.norm_cas(tra_ang+ casadi.SX([1e-8,0,0]))
        return theta,vector
    
    def magnit(self, ang):
        mag_sq = casadi.dot(ang, ang)
        mag = casadi.sqrt(mag_sq)
        return mag
    
    def norm_cas(self, ang):
        return ang/self.magnit(ang)
    
    def toQuaternion_casa(self,angle, dir):
        # if type(dir) == list:
        #     dir = numpy.array(dir)

        dir = self.norm_cas(dir)
        quat = casadi.SX.zeros(4)
        quat[0] = casadi.cos(angle / 2)
        quat[1:] = casadi.sin(angle / 2) * dir
        return quat


# ----------------------------------------------- For Animation ------------------------------------------------------#

    def play_animation(self, wing_len, state_traj, gate_traj1=None, gate_traj2=None,state_traj_ref=None, dt=0.01, \
            point1 = None,point2 = None,point3 = None,point4 = None,save_option=0, title='UAV Maneuvering'):
        font1 = {'family':'Times New Roman',
         'weight':'normal',
         'style':'normal', 'size':7}
        cm_2_inch = 2.54
        fig = plt.figure(figsize=(8/cm_2_inch,8*0.65/cm_2_inch),dpi=400)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', labelpad=-13,**font1)
        ax.set_ylabel('Y (m)', labelpad=-13,**font1)
        ax.set_zlabel('Z (m)', labelpad=-13,**font1)
        ax.tick_params(axis='x',which='major',pad=-5)
        ax.tick_params(axis='y',which='major',pad=-5)
        ax.tick_params(axis='z',which='major',pad=-5)
        ax.set_zlim(-5, 5)
        ax.set_ylim(-9, 9)
        ax.set_xlim(-6, 6)
        # ax.set_title(title, pad=20, fontsize=15)
        for t in ax.xaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(7)
        for t in ax.yaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(7)
        for t in ax.zaxis.get_major_ticks(): 
            t.label.set_font('Times New Roman') 
            t.label.set_fontsize(7)

        # target landing point
        ax.plot([self.goal_r_I[0]], [self.goal_r_I[1]], [self.goal_r_I[2]], c="r", marker="o",markersize=2)
        ax.view_init(25,-150)
        #plot the final state
        #final_position = self.get_final_position(wing_len=wing_len)
        #c_x, c_y, c_z = final_position[0:3]
        #r1_x, r1_y, r1_z = final_position[3:6]
        #r2_x, r2_y, r2_z = final_position[6:9]
        #r3_x, r3_y, r3_z = final_position[9:12]
        #r4_x, r4_y, r4_z = final_position[12:15]
        #line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=2, color='grey', marker='o', markersize=3)
        #line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=2, color='grey', marker='o', markersize=3)
        #line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=2, color='grey', marker='o', markersize=3)
        #line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=2, color='grey', marker='o', markersize=3)
        # plot gate
        if point1 is not None:
            ax.plot([point1[0],point2[0]],[point1[1],point2[1]],[point1[2],point2[2]],linewidth=1,color='red',linestyle='-')
            ax.plot([point2[0],point3[0]],[point2[1],point3[1]],[point2[2],point3[2]],linewidth=1,color='red',linestyle='-')
            ax.plot([point3[0],point4[0]],[point3[1],point4[1]],[point3[2],point4[2]],linewidth=1,color='red',linestyle='-')
            ax.plot([point4[0],point1[0]],[point4[1],point1[1]],[point4[2],point1[2]],linewidth=1,color='red',linestyle='-')
        # data
        position = self.get_quadrotor_position(wing_len, state_traj)
        sim_horizon = np.size(position, 0)

        if state_traj_ref is None:
            position_ref = self.get_quadrotor_position(0, numpy.zeros_like(position))
        else:
            position_ref = self.get_quadrotor_position(wing_len, state_traj_ref)

        ## plot the process of moving window and quadrotor
        #for i in range(10):
        #    a = i*6
        #    b = 0.9-0.1*i
        #    c = (b,b,b)
        #    c_x, c_y, c_z = position[a,0:3]
        #    r1_x, r1_y, r1_z = position[a,3:6]
        #    r2_x, r2_y, r2_z = position[a,6:9]
        #    r3_x, r3_y, r3_z = position[a,9:12]
        #    r4_x, r4_y, r4_z = position[a,12:15]
        #    line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=2, color=c, marker='o', markersize=3)
        #    line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=2, color=c, marker='o', markersize=3)
        #    line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=2, color=c, marker='o', markersize=3)
        #    line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=2, color=c, marker='o', markersize=3)

        #    p1_x, p1_y, p1_z = gate_traj1[a, 0,:]
        #    p2_x, p2_y, p2_z = gate_traj1[a, 1,:]
        #    p3_x, p3_y, p3_z = gate_traj1[a, 2,:]
        #    p4_x, p4_y, p4_z = gate_traj1[a, 3,:]
        #    gate_l1, = ax.plot([p1_x,p2_x],[p1_y,p2_y],[p1_z,p2_z],linewidth=1,color=c,linestyle='--')
        #    gate_l2, = ax.plot([p2_x,p3_x],[p2_y,p3_y],[p2_z,p3_z],linewidth=1,color=c,linestyle='--')
        #    gate_l3, = ax.plot([p3_x,p4_x],[p3_y,p4_y],[p3_z,p4_z],linewidth=1,color=c,linestyle='--')
        #    gate_l4, = ax.plot([p4_x,p1_x],[p4_y,p1_y],[p4_z,p1_z],linewidth=1,color=c,linestyle='--')
        

        ## animation
        # gate
        if gate_traj1 is not None:
            p1_x, p1_y, p1_z = gate_traj1[0, 0,:]
            p2_x, p2_y, p2_z = gate_traj1[0, 1,:]
            p3_x, p3_y, p3_z = gate_traj1[0, 2,:]
            p4_x, p4_y, p4_z = gate_traj1[0, 3,:]
            gate_l1, = ax.plot([p1_x,p2_x],[p1_y,p2_y],[p1_z,p2_z],linewidth=1,color='red',linestyle='-')
            gate_l2, = ax.plot([p2_x,p3_x],[p2_y,p3_y],[p2_z,p3_z],linewidth=1,color='red',linestyle='-')
            gate_l3, = ax.plot([p3_x,p4_x],[p3_y,p4_y],[p3_z,p4_z],linewidth=1,color='red',linestyle='-')
            gate_l4, = ax.plot([p4_x,p1_x],[p4_y,p1_y],[p4_z,p1_z],linewidth=1,color='red',linestyle='-')

            #p1_xa, p1_ya, p1_za = gate_traj2[0, 0,:]
            #p2_xa, p2_ya, p2_za = gate_traj2[0, 1,:]
            #p3_xa, p3_ya, p3_za = gate_traj2[0, 2,:]
            #p4_xa, p4_ya, p4_za = gate_traj2[0, 3,:]
            #gate_l1a, = ax.plot([p1_xa,p2_xa],[p1_ya,p2_ya],[p1_za,p2_za],linewidth=1,color='red',linestyle='--')
            #gate_l2a, = ax.plot([p2_xa,p3_xa],[p2_ya,p3_ya],[p2_za,p3_za],linewidth=1,color='red',linestyle='--')
            #gate_l3a, = ax.plot([p3_xa,p4_xa],[p3_ya,p4_ya],[p3_za,p4_za],linewidth=1,color='red',linestyle='--')
            #gate_l4a, = ax.plot([p4_xa,p1_xa],[p4_ya,p1_ya],[p4_za,p1_za],linewidth=1,color='red',linestyle='--')    

        # quadrotor
        line_traj, = ax.plot(position[:1, 0], position[:1, 1], position[:1, 2],linewidth=0.5)
        c_x, c_y, c_z = position[0, 0:3]
        r1_x, r1_y, r1_z = position[0, 3:6]
        r2_x, r2_y, r2_z = position[0, 6:9]
        r3_x, r3_y, r3_z = position[0, 9:12]
        r4_x, r4_y, r4_z = position[0, 12:15]
        line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=1, color='red', marker='o', markersize=1)
        line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=1, color='blue', marker='o', markersize=1)
        line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=1, color='orange', marker='o', markersize=1)
        line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=1, color='green', marker='o', markersize=1)

        line_traj_ref, = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='green', alpha=0.5)
        c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
        r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
        r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
        r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
        r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
        # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
        #                          color='green', marker='o', markersize=3, alpha=0.7)
        # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
        #                          color='green', marker='o', markersize=3, alpha=0.7)
        # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
        #                          color='green', marker='o', markersize=3, alpha=0.7)
        # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
        #                          color='green', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.2fs'
        time_text = ax.text2D(0.2, 0.7, "time", transform=ax.transAxes,**font1)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            line_traj.set_data(position[:num, 0], position[:num, 1])
            line_traj.set_3d_properties(position[:num, 2])


            # uav
            c_x, c_y, c_z = position[num, 0:3]
            r1_x, r1_y, r1_z = position[num, 3:6]
            r2_x, r2_y, r2_z = position[num, 6:9]
            r3_x, r3_y, r3_z = position[num, 9:12]
            r4_x, r4_y, r4_z = position[num, 12:15]

            line_arm1.set_data_3d([c_x, r1_x], [c_y, r1_y],[c_z, r1_z])
            #line_arm1.set_3d_properties()

            line_arm2.set_data_3d([c_x, r2_x], [c_y, r2_y],[c_z, r2_z])
            #line_arm2.set_3d_properties()

            line_arm3.set_data_3d([c_x, r3_x], [c_y, r3_y],[c_z, r3_z])
            #line_arm3.set_3d_properties()

            line_arm4.set_data_3d([c_x, r4_x], [c_y, r4_y],[c_z, r4_z])
            #line_arm4.set_3d_properties()

            # trajectory ref
            nu=sim_horizon-1
            line_traj_ref.set_data_3d(position_ref[:nu, 0], position_ref[:nu, 1],position_ref[:nu, 2])
            #line_traj_ref.set_3d_properties()

            # uav ref
            c_x_ref, c_y_ref, c_z_ref = position_ref[nu, 0:3]
            r1_x_ref, r1_y_ref, r1_z_ref = position_ref[nu, 3:6]
            r2_x_ref, r2_y_ref, r2_z_ref = position_ref[nu, 6:9]
            r3_x_ref, r3_y_ref, r3_z_ref = position_ref[nu, 9:12]
            r4_x_ref, r4_y_ref, r4_z_ref = position_ref[nu, 12:15]

            # line_arm1_ref.set_data_3d([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref],[c_z_ref, r1_z_ref])
            # #line_arm1_ref.set_3d_properties()

            # line_arm2_ref.set_data_3d([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref],[c_z_ref, r2_z_ref])
            # #line_arm2_ref.set_3d_properties()

            # line_arm3_ref.set_data_3d([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref],[c_z_ref, r3_z_ref])
            # #line_arm3_ref.set_3d_properties()

            # line_arm4_ref.set_data_3d([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref],[c_z_ref, r4_z_ref])
            #line_arm4_ref.set_3d_properties()

            ## plot moving gate
            if gate_traj1 is not None:
                p1_x, p1_y, p1_z = gate_traj1[num, 0,:]
                p2_x, p2_y, p2_z = gate_traj1[num, 1,:]
                p3_x, p3_y, p3_z = gate_traj1[num, 2,:]
                p4_x, p4_y, p4_z = gate_traj1[num, 3,:]       

                gate_l1.set_data_3d([p1_x,p2_x],[p1_y,p2_y],[p1_z,p2_z])
                gate_l2.set_data_3d([p2_x,p3_x],[p2_y,p3_y],[p2_z,p3_z]) 
                gate_l3.set_data_3d([p3_x,p4_x],[p3_y,p4_y],[p3_z,p4_z]) 
                gate_l4.set_data_3d([p4_x,p1_x],[p4_y,p1_y],[p4_z,p1_z])


                #p1_xa, p1_ya, p1_za = gate_traj2[num, 0,:]
                #p2_xa, p2_ya, p2_za = gate_traj2[num, 1,:]
                #p3_xa, p3_ya, p3_za = gate_traj2[num, 2,:]
                #p4_xa, p4_ya, p4_za = gate_traj2[num, 3,:]       

                #gate_l1a.set_data_3d([p1_xa,p2_xa],[p1_ya,p2_ya],[p1_za,p2_za])
                #gate_l2a.set_data_3d([p2_xa,p3_xa],[p2_ya,p3_ya],[p2_za,p3_za]) 
                #gate_l3a.set_data_3d([p3_xa,p4_xa],[p3_ya,p4_ya],[p3_za,p4_za]) 
                #gate_l4a.set_data_3d([p4_xa,p1_xa],[p4_ya,p1_ya],[p4_za,p1_za])




                return line_traj,gate_l1,gate_l2,gate_l3,gate_l4,line_arm1, line_arm2, line_arm3, line_arm4, \
                    line_traj_ref, time_text
                                            #, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref
            return line_traj, line_arm1, line_arm2, line_arm3, line_arm4, \
                line_traj_ref, time_text #, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*500, blit=True)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2'+title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def plot_position(self,state_traj,dt = 0.1):
        fig, axs = plt.subplots(3)
        fig.suptitle('position vs t')
        N = len(state_traj[:,0])
        x = np.arange(0,N*dt,dt)
        axs[0].plot(x,state_traj[:,0])
        axs[1].plot(x,state_traj[:,1])
        axs[2].plot(x,state_traj[:,2])
        plt.show()
        
    def plot_velocity(self,state_traj,dt = 0.1):
        fig, axs = plt.subplots(3)
        fig.suptitle('velocity vs t')
        N = len(state_traj[:,0])
        x = np.arange(0,N*dt,dt)
        axs[0].plot(x,state_traj[:,3])
        axs[1].plot(x,state_traj[:,4])
        axs[2].plot(x,state_traj[:,5])
        plt.show()

    def plot_quaternions(self,state_traj,dt = 0.1):
        fig, axs = plt.subplots(4)
        fig.suptitle('quaternions vs t')
        N = len(state_traj[:,0])
        x = np.arange(0,N*dt,dt)
        axs[0].plot(x,state_traj[:,6])
        axs[1].plot(x,state_traj[:,7])
        axs[2].plot(x,state_traj[:,8])
        axs[3].plot(x,state_traj[:,9])
        plt.show()
    
    def plot_angularrate(self,state_traj,dt = 0.01):
        plt.title('angularrate vs time')
        N = len(state_traj[:,0])
        x = np.arange(0,N*dt,dt)
        plt.plot(x,state_traj[:,10],color = 'b', label = 'w1')
        plt.plot(x,state_traj[:,11],color = 'r', label = 'w2')
        plt.plot(x,state_traj[:,12],color = 'y', label = 'w3')
        plt.xlabel('t')
        plt.ylabel('w')
        plt.grid(True,color='0.6',dashes=(2,2,1,1))
        plt.legend()
        plt.savefig('./angularrate.png')
        plt.show()
        

    def plot_input(self,control_traj,dt = 0.1):
        N = int(len(control_traj[:,0]))
        x = np.arange(0,round(N*dt,1),dt)
        plt.plot(x,control_traj[:,0],color = 'b', label = 'u1')
        plt.plot(x,control_traj[:,1],color = 'r', label = 'u2')
        plt.plot(x,control_traj[:,2],color = 'y', label = 'u3')
        plt.plot(x,control_traj[:,3],color = 'g', label = 'u4')
        plt.title('input vs time')
        plt.ylim([0,5])
        plt.xlabel('t')
        plt.ylabel('u')
        plt.grid(True,color='0.6',dashes=(2,2,1,1))
        plt.legend()
        plt.savefig('./input.png')
        plt.show()
        

    def plot_T(self,control_traj,dt = 0.1):
        N = int(len(control_traj[:,0]))
        x = np.arange(0,round(N*dt,1),dt)
        plt.plot(x,control_traj[:,0],color = 'b', label = 'T')
        plt.title('input vs time')
        plt.ylim([0,20])
        plt.xlabel('t')
        plt.ylabel('T')
        plt.grid(True,color='0.6',dashes=(2,2,1,1))
        plt.legend()
        plt.savefig('./input_T.png')
        plt.show()
        
    
    def plot_M(self,control_traj,dt = 0.1):
        N = int(len(control_traj[:,0]))
        x = np.arange(0,round(N*dt,1),dt)
        plt.plot(x,control_traj[:,1],color = 'r', label = 'Mx')
        plt.plot(x,control_traj[:,2],color = 'y', label = 'My')
        plt.plot(x,control_traj[:,3],color = 'g', label = 'Mz')
        plt.title('input vs time')
        plt.ylim([0,1])
        plt.xlabel('t')
        plt.ylabel('T')
        plt.grid(True,color='0.6',dashes=(2,2,1,1))
        plt.legend()
        plt.savefig('./input_M.png')
        plt.show()
        



    def dir_cosine(self, q): # world frame to body frame
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )

## define the class of the gate (kinematics)
class gate:
    ## using 12 coordinates to define a gate
    def __init__(self, gate_point = None):
        self.gate_point = gate_point

        ##obtain the position (centroid)
        self.centroid = np.array([np.mean(self.gate_point[:,0]),np.mean(self.gate_point[:,1]),np.mean(self.gate_point[:,2])])

        ## obtain the orientation using the unit vector in the world frame
        az = norm(np.array([0,0,1]))
        ay = norm(np.cross(self.gate_point[1]-self.gate_point[0],self.gate_point[2]-self.gate_point[1]))
        ax = np.cross(ay,az)
        self.ay = ay
        self.I_G = np.array([ax,ay,az]).T # rotaton matrix from the world frame to the gap-attached frame

    ## rotate an angle around y axis of thw window
    def rotate_y(self,angle):
        ## define the rotation matrix to rotate
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        gate_point = self.gate_point - np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        for i in range(4):
            [gate_point[i,0],gate_point[i,2]] = np.matmul(rotation,np.array([gate_point[i,0],gate_point[i,2]]))
        self.gate_point = gate_point + np.array([self.centroid,self.centroid,self.centroid,self.centroid])

        ## update the orientation and the position
        self.centroid = np.array([np.mean(self.gate_point[:,0]),np.mean(self.gate_point[:,1]),np.mean(self.gate_point[:,2])])
        az = norm(np.array([0,0,1]))
        ay = norm(np.cross(self.gate_point[1]-self.gate_point[0],self.gate_point[2]-self.gate_point[1]))
        ax = np.cross(ay,az)
        self.ay = ay
        self.I_G = np.array([ax,ay,az]) # rotation matrix from gate frame to world frame

    ## rotate an angle around z axis of thw window
    def rotate(self,angle):
        ## define the rotation matrix to rotate
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        gate_point = self.gate_point - np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        for i in range(4):
            gate_point[i,0:2] = np.matmul(rotation,gate_point[i,0:2])
        self.gate_point = gate_point + np.array([self.centroid,self.centroid,self.centroid,self.centroid])

        ## update the orientation and the position
        self.centroid = np.array([np.mean(self.gate_point[:,0]),np.mean(self.gate_point[:,1]),np.mean(self.gate_point[:,2])])
        az = norm(np.array([0,0,1]))
        ay = norm(np.cross(self.gate_point[1]-self.gate_point[0],self.gate_point[2]-self.gate_point[1]))
        ax = np.cross(ay,az)
        self.ay = ay
        self.I_G = np.array([ax,ay,az])

    ## translate the gate in world frame
    def translate(self,displace):
        self.gate_point = self.gate_point + np.array([displace,displace,displace,displace])
        self.centroid = np.array([np.mean(self.gate_point[:,0]),np.mean(self.gate_point[:,1]),np.mean(self.gate_point[:,2])])

        ## update the orientation and the positio
        az = norm(np.array([0,0,1]))
        ay = norm(np.cross(self.gate_point[1]-self.gate_point[0],self.gate_point[2]-self.gate_point[1]))
        ax = np.cross(ay,az)
        self.ay = ay
        self.I_G = np.array([ax,ay,az]) # this is a rotation matrix from gate frame to inertial frame, which is an identity matrix.

    ## 'out' means return the 12 coordinates of the gate
    def translate_out(self,displace):
        return self.gate_point + np.array([displace,displace,displace,displace])

    def rotate_y_out(self,angle):
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        gate_point = self.gate_point - np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        for i in range(4):
            [gate_point[i,0],gate_point[i,2]] = np.matmul(rotation,np.array([gate_point[i,0],gate_point[i,2]]))
        gate_point = gate_point + np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        return gate_point

    def rotate_out(self,angle):
        rotation = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        gate_point = self.gate_point - np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        for i in range(4):
            gate_point[i,0:2] = np.matmul(rotation,gate_point[i,0:2])
        gate_point = gate_point + np.array([self.centroid,self.centroid,self.centroid,self.centroid])
        return gate_point

    ## given time horizon T and time interval dt, return a sequence of position representing the random move of the gate
    # def random_move(self, T = 4, dt = 0.01):
    #     gate_point = self.gate_point
    #     move = [gate_point]
    #     ## initial random velocity
    #     velo = np.random.normal(0,0.2,size=2)
    #     for i in range(int(T/dt)):
    #         ## random acceleration
    #         accel = np.random.normal(0,2,size=2)
    #         ## integration
    #         velo += dt*accel
    #         velocity = np.clip(np.array([velo[0],0,velo[1]]),-0.4,0.4)
    #         for j in range(4):
    #             gate_point[j] += dt * velocity
    #         move = np.concatenate((move,[gate_point]),axis=0)
    #     return move
    
    ## given constant velocity and angular velocity around y axis, return a sequence of position representing the random move of the gate 
    def move(self, T = 5, dt = 0.01, v = [0,0,0], w = 0):
        gate_point = self.gate_point
        move = [gate_point]
        velo = np.array(v) 
        V    = [velo]
        
        # define the rotation matrix
        rotation = np.array([[math.cos(dt*w),-math.sin(dt*w)],[math.sin(dt*w),math.cos(dt*w)]])
        for i in range(int(T/dt)):
            v_noise = np.clip(np.random.normal(0,0.1,3),-0.1,0.1)
            centroid = np.array([np.mean(gate_point[:,0]),np.mean(gate_point[:,1]),np.mean(gate_point[:,2])])
            gate_pointx = gate_point - np.array([centroid,centroid,centroid,centroid]) # coordinates in the window body frame
            # rotation about the y axis
            for i in range(4):
                [gate_pointx[i,0],gate_pointx[i,2]] = np.matmul(rotation,np.array([gate_pointx[i,0],gate_pointx[i,2]]))
            gate_point = gate_pointx + np.array([centroid,centroid,centroid,centroid])
            # translation
            for j in range(4):
                gate_point[j] += dt * (velo+v_noise)
            move = np.concatenate((move,[gate_point]),axis=0)
            V    = np.concatenate((V,[velo+v_noise]),axis=0)
        return move, V
    
    ## transform the state in world frame to the state in window frame
    def transform(self, inertial_state):
        outputs = np.zeros(13)
        ## position
        outputs[0:3] = np.matmul(self.I_G, inertial_state[0:3] - self.centroid) # relative position, the future gap is viewed to be static
        ## velocity
        outputs[3:6] = np.matmul(self.I_G, inertial_state[3:6])
        ## angular velocity
        outputs[10:13] = inertial_state[10:13]
        ## attitude
        quat = np.zeros(4)
        quat[0:3] = inertial_state[7:10]
        quat[3] = inertial_state[6]
        r1 = R.from_quat(quat)
        # attitude transformation
        r2 = R.from_matrix(np.matmul(self.I_G,r1.as_matrix()))
        quat_out = np.array(r2.as_quat())
        outputs[6] = quat_out[3]
        outputs[7:10] = quat_out[0:3]
        return outputs

    ## transform the final point in world frame to the point in window frame
    def t_final(self, final_point):
        return np.matmul(self.I_G, final_point - self.centroid)
        

def toQuaternion(angle, dir):
    if type(dir) == list:
        dir = numpy.array(dir)
    dir = dir / numpy.linalg.norm(dir)
    quat = numpy.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat.tolist()


# normalized verctor
def normalizeVec(vec):
    if type(vec) == list:
        vec = np.array(vec)
    vec = vec / np.linalg.norm(vec)
    return vec


def quaternion_conj(q):
    conj_q = q
    conj_q[1] = -q[1]
    conj_q[2] = -q[2]
    conj_q[3] = -q[3]
    return conj_q
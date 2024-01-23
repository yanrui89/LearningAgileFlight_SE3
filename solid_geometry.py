# solid geometry
# this file is to do some calculation of solid geometry to do the collision detection of quadrotor
# this file consists of several classes
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

## return the maginitude of a vector
def magni(vector):
    return np.sqrt(np.dot(np.array(vector),np.array(vector)))

## return the unit vector of a vector
def norm(vector):
    return np.array(vector)/magni(np.array(vector))


## define a class of a plane (using three points on the plane)
class plane():
    def __init__(self, point1, point2, point3):
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.point3 = np.array(point3)
        self.vec1 = self.point2 - self.point1
        self.vec2 = self.point3 - self.point1
        self.normal = norm(np.cross(self.vec2,self.vec1))
    
    # normal vector of the plane 
    def nor_vec(self, ):
        return self.normal

    # normal vector of one side 
    def n1(self):
        return norm(np.cross(self.vec1,self.normal))

    # normal vector of one side 
    def n2(self):
        return norm(np.cross(self.normal,self.vec2))

    # normal vector of one side 
    def n3(self):
        self.vec3 = self.point3 - self.point2
        return norm(np.cross(self.normal,self.vec3))
    
    ## intersection with another line 
    def interpoint(self, point1, point2):
        dir = norm(np.array(point1)-np.array(point2))
        t = 1/(np.dot(dir,self.normal))*(np.dot(self.normal,np.array(point1)-self.point1))
        point = np.array(point1) - t * dir
        return point

## define a class of a line
class line():
    def __init__(self, point1, point2):
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.dir = norm(self.point1 - self.point2)

    ## return the distance from a point to the line
    def vertical(self, point):
        point3 = np.array(point)
        normal = np.cross(point3 - self.point1, self.dir)
        return magni(normal)

    ## return the distance from a point to the line section
    def distance(self,point):
        a = self.vertical(point)
        b = magni(point-self.point1)
        c = magni(point-self.point2)
        d = magni(self.point1-self.point2)
        if(b>c):
            if((b**2-d**2)>a**2):
                dis = c
            else:
                dis = a
        else:
            if((c**2-d**2)>a**2):
                dis = b
            else:
                dis = a
        return dis

## define the narrow window which is also the obstacle for the quadrotor
class obstacle():
    def __init__(self, point1, point2, point3, point4):
        self.point1 = np.array(point1)
        self.point2 = np.array(point2)
        self.point3 = np.array(point3)
        self.point4 = np.array(point4)

        #define the centroid of obstacle
        self.centroid = np.array([(point1[0]+point2[0]+point3[0]+point4[0])/4,\
            (point1[1]+point2[1]+point3[1]+point4[1])/4,(point1[2]+point2[2]+point3[2]+point4[2])/4])
        
        #define the cross section
        self.plane1 = plane(self.centroid,point1,point2)
        self.plane2 = plane(self.centroid,point2,point3)
        self.plane3 = plane(self.centroid,point3,point4)
        self.plane4 = plane(self.centroid,point4,point1)

        #define the bound
        self.line1 = line(point1, point2)
        self.line2 = line(point2, point3)
        self.line3 = line(point3, point4)
        self.line4 = line(point4, point1)

    def collis_det(self, vert_traj, horizon):
        ## define the state whether find corresponding plane
        collision = 0
        self.co = 0

        ## judge if the trajectory traverse through the plane
        if((np.dot(self.plane1.nor_vec(),vert_traj[0]-self.centroid)<0)):
            return 0

        ## judge whether the first plane is the traversal plane
        # find two points of traverse
        d_min = 0.2
        for t in range(horizon):
            if(np.dot(self.plane1.nor_vec(),vert_traj[t]-self.centroid)<0):
                intersect = self.plane1.interpoint(vert_traj[t],vert_traj[t-1])
                # judge whether they belong to plane1 and calculate the distance
                if(np.dot(self.plane1.n1(),intersect-self.centroid)>0 and np.dot(self.plane1.n2(),intersect-self.centroid)>0):
                    if(np.dot(self.point1-intersect,self.plane1.n3())>0):
                        m = min(self.line1.vertical(intersect),self.line2.vertical(intersect),\
                            self.line3.vertical(intersect),self.line4.vertical(intersect))
                        collision = - max(0,d_min-m)**2
                        self.co = 1
                    else:
                        m = min(self.line4.distance(intersect),self.line1.distance(intersect),self.line2.distance(intersect))
                        collision =   - 2*d_min*m - d_min**2
                

       
                # judge whether the intersection belongs to plane2 and calculate the distance  
                if(np.inner(self.plane2.n1(),intersect-self.centroid)>0 and np.inner(self.plane2.n2(),intersect-self.centroid)>0):
                    if(np.dot(self.point2-intersect,self.plane2.n3())>0):
                        m = min(self.line1.vertical(intersect),self.line2.vertical(intersect),\
                            self.line3.vertical(intersect),self.line4.vertical(intersect))
                        collision = - max(0,d_min-m)**2
                        self.co = 1
                    else:
                        m = min(self.line1.distance(intersect),self.line2.distance(intersect),self.line3.distance(intersect))
                        collision =   - 2*d_min*m - d_min**2
                    

                # judge whether the intersection belongs to plane3 and calculate the distance
                if(np.inner(self.plane3.n1(),intersect-self.centroid)>0 and np.inner(self.plane3.n2(),intersect-self.centroid)>0):
                    if(np.dot(self.point3-intersect,self.plane3.n3())>0):
                        m = min(self.line1.vertical(intersect),self.line2.vertical(intersect),\
                            self.line3.vertical(intersect),self.line4.vertical(intersect))
                        collision = - max(0,d_min-m)**2
                        self.co = 1
                    else:
                        m = min(self.line2.distance(intersect),self.line3.distance(intersect),self.line4.distance(intersect))
                        collision =   - 2*d_min*m - d_min**2
                    

                # judge whether the intersection belongs to plane4 and calculate the distance
                if(np.inner(self.plane4.n1(),intersect-self.centroid)>0 and np.inner(self.plane4.n2(),intersect-self.centroid)>0):
                    if(np.dot(self.point4-intersect,self.plane4.n3())>0):
                        m = min(self.line1.vertical(intersect),self.line2.vertical(intersect),\
                            self.line3.vertical(intersect),self.line4.vertical(intersect))
                        collision = - max(0,d_min-m)**2
                        self.co = 1
                    else:
                        m = min(self.line3.distance(intersect),self.line4.distance(intersect),self.line1.distance(intersect))
                        collision =   - 2*d_min*m - d_min**2
                break
                        
        return collision
    
class obstacleNewReward(obstacle):
    def __init__(self, point1, point2, point3, point4, wingrad, alpha, beta, gamma):
        super().__init__(point1 = point1,
                         point2 = point2,
                         point3 = point3,
                         point4 = point4)              
        
        self.wingrad = wingrad  
        self.height = 0.3
        self.scaledMatrix = np.diag([self.wingrad, self.wingrad, self.height])    

        self.midpoint1 = (self.point1 + self.point2)/2
        self.midpoint2 = (self.point2 + self.point3)/2
        self.midpoint3 = (self.point3 + self.point4)/2
        self.midpoint4 = (self.point4 + self.point1)/2

        self.R_max = 100
        self.gamma = gamma #10
        self.alpha = alpha #1
        self.beta = beta #100
         
    def collis_det(self, vert_traj, horizon, quat, trav_t):
        ## define the state whether find corresponding plane
        self.co = 0

        trav_idx = trav_t / 0.1
        pre_idx = int(np.floor(trav_idx))
        # post_idx = np.floor(trav_idx)
        # intersect = self.plane1.interpoint(vert_traj[t],vert_traj[t-1])
        X = vert_traj[pre_idx]
        # judge whether they belong to plane1 and calculate the distance
        curr_quat = quat[pre_idx,:]

        col_reward, curr_col1, curr_col2, self.co, curr_delta1, curr_delta2, curr_delta3, curr_delta4 = self.collis_det_ep(X, curr_quat)

                        
        return col_reward, curr_col1, curr_col2, self.co, curr_delta1, curr_delta2, curr_delta3, curr_delta4
    
    def collis_det_ep(self, X, curr_quat):
  
        # post_idx = np.floor(trav_idx)
        # intersect = self.plane1.interpoint(vert_traj[t],vert_traj[t-1])
        intersect = X
        # judge whether they belong to plane1 and calculate the distance
        # curr_r_I2B2 = R.from_quat(np.array([curr_quat])).as_matrix().squeeze()
        # curr_r_I2B = self.quaternion_rotation_matrix(curr_quat)
        test_quat = np.array([curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]])
        curr_r_I2B = self.quaternion_rotation_matrix(test_quat)
        # test = self.dir_cosine(curr_quat)
        # curr_r_I2B_chk = self.quaternion_rotation_matrix(curr_quat)
        E = np.matmul(self.scaledMatrix, curr_r_I2B.transpose())
        E = np.matmul(curr_r_I2B, E)

        # mat_inverse = np.linalg.inv(E)
        # mat_inverse2 = self.matrix_inverse(E)


        curr_delta1 = np.linalg.norm(np.matmul(np.linalg.inv(E), self.midpoint1 - intersect))
        curr_delta11 = np.linalg.norm(np.matmul(np.linalg.inv(E), self.point1 - intersect))
        curr_delta2 = np.linalg.norm(np.matmul(np.linalg.inv(E), self.midpoint2 - intersect))
        curr_delta22 = np.linalg.norm(np.matmul(np.linalg.inv(E), self.point2 - intersect))
        curr_delta3 = np.linalg.norm(np.matmul(np.linalg.inv(E), self.midpoint3 - intersect))
        curr_delta33 = np.linalg.norm(np.matmul(np.linalg.inv(E), self.point3 - intersect))
        curr_delta4 = np.linalg.norm(np.matmul(np.linalg.inv(E), self.midpoint4 - intersect))
        curr_delta44 = np.linalg.norm(np.matmul(np.linalg.inv(E), self.point4 - intersect))

        if curr_delta1 < 1:
            self.co = 1
        if curr_delta2 < 1:
            self.co = 1
        if curr_delta3 < 1:
            self.co = 1
        if curr_delta4 < 1:
            self.co = 1
        if curr_delta11 < 1:
            self.co = 1
        if curr_delta22 < 1:
            self.co = 1
        if curr_delta33 < 1:
            self.co = 1
        if curr_delta44 < 1:
            self.co = 1
            
        curr_col1 = (self.gamma  * np.power(curr_delta1, 2)) + (self.gamma  * np.power(curr_delta2, 2)) + (self.gamma  * np.power(curr_delta3, 2)) + (self.gamma  * np.power(curr_delta4, 2)) + (self.gamma  * np.power(curr_delta11, 2)) + (self.gamma  * np.power(curr_delta22, 2)) + (self.gamma  * np.power(curr_delta33, 2)) + (self.gamma  * np.power(curr_delta44, 2))
        curr_col2 = (self.alpha * np.exp(self.beta*(1 - curr_delta1))) + (self.alpha * np.exp(self.beta*(1 - curr_delta2))) + (self.alpha * np.exp(self.beta*(1 - curr_delta3))) + (self.alpha * np.exp(self.beta*(1 - curr_delta4))) + (self.alpha * np.exp(self.beta*(1 - curr_delta11))) + (self.alpha * np.exp(self.beta*(1 - curr_delta22))) + (self.alpha * np.exp(self.beta*(1 - curr_delta33))) + (self.alpha * np.exp(self.beta*(1 - curr_delta44)))

        collision = curr_col1 + curr_col2

                        
        return self.R_max - collision, curr_col1, curr_col2, self.co, curr_delta1, curr_delta2, curr_delta3, curr_delta4
    
    def quaternion_rotation_matrix(self,Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        """
        # mod_Q = np.sqrt(Q[0]*Q[0] + Q[1]*Q[1] + Q[2]*Q[2] + Q[3]*Q[3])
        Q = Q / np.linalg.norm(Q)
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
        
        # First row of the rotation matrix
        r00 = 1 - 2*q1*q1 - 2*q2*q2
        r01 = 2*q0*q1 - 2*q2*q3
        r02 = 2*q0*q2 + 2*q1*q3
        
        # Second row of the rotation matrix
        r10 = 2*q0*q1 + 2*q2*q3 
        r11 = 1 - 2*q0*q0 - 2*q2*q2
        r12 = 2*q1*q2 - 2*q0*q3
        
        # Third row of the rotation matrix
        r20 = 2*q0*q2 - 2*q3*q1
        r21 = 2*q1*q2 + 2*q3*q0
        r22 = 1 - 2*q0*q0 - 2*q1*q1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])
                                
        return rot_matrix
    
    def dir_cosine(self, q): # world frame to body frame
        q = q / np.linalg.norm(q)
        C_B_I = np.vstack((
            np.hstack((1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2]))),
            np.hstack((2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1]))),
            np.hstack((2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        ))
        return C_B_I
    

    def matrix_minor(self, matrix, row, col):
        """Calculate the minor of a matrix by excluding the specified row and column."""
        return [[matrix[i][j] for j in range(len(matrix[i])) if j != col] for i in range(len(matrix)) if i != row]

    def matrix_determinant(self,matrix):
        """Calculate the determinant of a 2x2 matrix."""
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    def matrix_inverse(self, matrix):
        """Calculate the inverse of a 3x3 matrix."""
        det = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        print(det)
        if det == 0:
            raise ValueError("Matrix is not invertible.")

        cofactors = [
            [self.matrix_determinant(self.matrix_minor(matrix, i, j)) * ((-1) ** (i + j)) for j in range(3)]
            for i in range(3)
        ]

        adjugate = [[cofactors[j][i] for j in range(3)] for i in range(3)]

        inverse = [[adjugate[i][j] / det for j in range(3)] for i in range(3)]

        return inverse

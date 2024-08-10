import sys
import os 
import pandas as pd
import time
import numpy as np
from gz.math7 import Quaterniond, Vector3d
import matplotlib.pyplot as plt
import csv
import time
import sympy as sym

SOURCE_FOLDER = os.path.dirname(os.path.abspath(__file__))

BENCHMARK_NAME = sys.argv[1]

MODEL_NAME = ['30', '45', '60', '75', '90']
I_30 = [[],[],[]]
I_45 = [[],[],[]]
I_60 = [[],[],[]]
I_75 = [[],[],[]]
I_90 = [[],[],[]]

class postProcessing:

    def __init__(self, test_name):

        self.mass = {'30': 0.05, '45': 0.06, '60': 0.04, 
                     '75': 0.076, '90': 0.0604}
        self.com = {'30': 0.05, '45': 0.06, '60': 0.04, 
                     '75': 0.076, '90': 0.0604}
        
        self.I = {'30': np.array(I_30), '45': np.array(I_45), 
                  '60': np.array(I_60), '75': np.array(I_75),
                  '90': np.array(I_90)}
        self.mu = 0.9

        self.fdir1 = np.array([1, 0, 0])
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        metrics_filename = test_name + "_" + timestr + ".csv"
        self.metrics_path = os.path.join(SOURCE_FOLDER, "test_results", metrics_filename)
        print(f"metrics path is {self.metrics_path}")

        self.csv_file = open(self.metrics_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        metrics = ["angMomentum0", "angMomentumErr_maxAbs","angPositionErr_x_maxAbs", 
           "angPositionErr_y_maxAbs", "angPositionErr_z_maxAbs", "collision", 
           "dt", "energyError_maxAbs", 	"engine", "isComplex", "linPositionErr_maxAbs",
           "linVelocityErr_maxAbs", "modelCount", "simTime", "time", "timeRatio", "classname"]
        self.csv_writer.writerow(metrics)

    def read_file(self,file_path: str):
        benchmark_config = pd.read_csv(file_path, nrows=1).to_numpy()
        states = pd.read_csv(file_path,skiprows=2).to_numpy()
        return benchmark_config, states

    def get_file_names(self):
        '''Method to obtain the file names and file paths of benchmark result'''
        result_dir = os.path.join(SOURCE_FOLDER, "test_results", BENCHMARK_NAME, "CSV")
        file_names = os.listdir(result_dir) 
        return result_dir, file_names
    
    def set_test_parameters(self, physic_engine, dt, friction_model,
                            complex, surface_slope, friction_coefficient, 
                            cog_height, equal_ke, computation_time):
        self.physics_engine = physic_engine
        self.dt = dt
        self.fricition_model = friction_model
        self.complex = complex
        self.surface_slope = surface_slope
        self.friction_coefficient = friction_coefficient
        self.cog_height = cog_height
        self.equal_ke = equal_ke
        self.computation_time = computation_time

        if complex:
            self.class_name = "static"
        else:
            self.class_name = "sliding"
    
    def hat(v):
        # hat operator for vector
        v = v.ravel()
        return np.array([[0, v[2], -v[1]],
                         [-v[2], 0, v[0]],
                         [v[1], -v[0], 0]])

    def compute_pos_mass(self, face_angle: float, model_no: int):
        h = 0.15
        k = 0.05
        rho = 600
        r_c = 0.005
        r_s = 0.02

        ## centre of gravity location of ball and rod
        self.ball_pos = {'a': np.array([[h - k , (model_no-1), 0.02]]),
                    'b': np.array([[- k , h*np.tan(face_angle) + (model_no - 1), 0.02]]),
                    'c': np.array([[- k , - h*np.tan(face_angle) + (model_no -1), 0.02]])}
        
        v_s = (4*np.pi*r_s**3)/3
        self.m_s = rho*v_s
        
        self.rod_pos = {'ab': (self.ball_pos['a'] + self.ball_pos['b'])/2,
                   'bc': (self.ball_pos['b'] + self.ball_pos['c'])/2, 
                   'ca': (self.ball_pos['c'] + self.ball_pos['a'])/2}
        
        self.l_ab = np.linalg.norm(self.ball_pos['b'] - self.ball_pos['a'])
        self.l_bc = np.linalg.norm(self.ball_pos['c'] - self.ball_pos['b'])
        self.l_ca = np.linalg.norm(self.ball_pos['a'] - self.ball_pos['c'])
        
        self.m_r1 = np.pi*(r_c**2)*self.l_ab*rho
        self.m_r2 = np.pi*(r_c**2)*self.l_bc*rho
        self.m_r3 = np.pi*(r_c**2)*self.l_ca*rho

    def compute_contact_force(self, slope: float, model_no: int, 
                              contact_pos_a, contact_pos_b, contact_pos_c):
        
        F_a = np.array([sym.symbols('f_a%d' % i) for i in range(3)]).reshape(3,1)
        F_b = np.array([sym.symbols('f_b%d' % i) for i in range(3)]).reshape(3,1)
        F_c = np.array([sym.symbols('f_c%d' % i) for i in range(3)]).reshape(3,1)

        g = np.array([-np.sin(slope)*9.8,0,-np.cos(slope)*9.8]).reshape(3,1)

        eq_f = F_a + F_b + F_c + (3*self.m_s + self.m_r1 + self.m_r2 + self.m_r3)*g

        # z direction force balance
        eq1 = eq_f[0][0]
        eq2 = eq_f[1][0]
        eq3 = eq_f[2][0]
        
        p_a = self.ball_pos['a']
        p_b = self.ball_pos['b']
        p_c = self.ball_pos['c']

        p_ab = self.rod_pos['ab']
        p_bc = self.rod_pos['bc']
        p_ca = self.rod_pos['ca']
    
        # moment balance about A
        eq_t = self.hat(p_ab - p_a) @ (self.m_r1 * g) + self.hat(p_b - p_a) @ (self.m_s * g + F_b) +\
               self.hat(p_bc - p_a) @ (self.m_r2 * g) + self.hat(p_c - p_a) @ (self.m_s *g + F_c) +\
               self.hat(p_ca - p_a) @ (self.m_r3 * g)
        
        eq4 = eq_t[3][0]
        eq5 = eq_t[4][0]
        eq6 = eq_t[5][0]
        eq7 = eq_t[6][0]
        eq8 = eq_t[7][0]
        eq9 = eq_t[8][0]

        result = sym.linsolve([eq1, eq2, eq3, eq4, 
                               eq5, eq6, eq7, eq8, eq9],
                              (F_a[0], F_a[1], F_a[2],
                               F_b[0], F_b[1], F_b[2],
                               F_c[0], F_c[1], F_c[2],))
        
        return  list(result.args)
    
    def compute_slip_vel(self, com_vel: np.ndarray,
                         ang_vel: np.ndarray, r: np.ndarray):
        slip_vel = com_vel + self.hat(r) @ ang_vel
        return slip_vel
        
        

    
    def calculate_mertics(self, states: np.ndarray,
                          model_no: int, face_angle: str,
                          slope: float):
        sim_time = states[:, 0]
        linear_accel = states[:, 2:5]
        angular_accel = states[:, 5:8]
        v = states[:, 8:11]
        omega = states[:, 11,14]
        pos = states[:, 14:17]  
        rot = states[:, 17:21]
        contact1_info = states[:, 21:34]      
        contact2_info = states[:, 34:47]
        contact3_info = states[:, 47:]

        g = np.array([-np.sin(slope)*9.8,0,-np.cos(slope)*9.8]).reshape(3,1)

        E = np.zeros(self.N)
        F_mag_error = np.zeros((self.N,3))
        F_dir_error = np.zeros((self.N,3))
        L = np.zeros((self.N,3))
        
        for i in range(self.N):
            # angular velocity in body frame
            omega_w = omega[i].tolist()
            quat = rot[i].tolist()
            quat = Quaterniond(quat[0], quat[1], quat[2], quat[3])
            omega_b = quat.rotate_vector_reverse(Vector3d(omega_w[0], omega_w[1], omega_w[2]))
            omega_b = np.array([omega_b[0], omega_b[1], omega_b[2]])

            # translation energy + rotational energy + potential energy
            tran_E = 0.5*self.m*v[i].dot(v[i])
            rot_E = 0.5*omega_b.dot(self.I.dot(omega_b))
            V = - self.m*self.gravity.dot(pos[i])
            E[i] = tran_E + rot_E + V

            # angular momentum in body frame 
            l_b = self.I.dot(omega_b).tolist()

            # angular momentum in world frame
            l_vector =  Vector3d(l_b[0], l_b[1],l_b[2])
            l_w = quat.rotate_vector(l_vector)
            L[i] = np.array([l_w[0], l_w[1], l_w[2]])

            if contact1_info[i,0]:
                normal = contact1_info[i, 4:7]
                force = contact1_info[i, 7:10]
                
                ## contact force along normal (contact_force.normal)
                force_n = np.dot(force, normal)

                r_ac = contact1_info[i, 1:4] - pos[i]
                slip_vel = self.compute_slip_vel(v[i], omega[i], r_ac)
                ## analytical solution
                friction_force_a  = self.mu * force_n * (slip_vel/np.linalg.norm(slip_vel))
                ## numerical solution
                friciton_force_n = contact1_info[i,7:9]
                F_mag_error[i,0] = (friciton_force_n - friction_force_a)[:2]




        initial_energy = E[0]

        enery_error = E - initial_energy
        




    




     
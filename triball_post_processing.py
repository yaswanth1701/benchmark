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

I_30 = [[8.4732487219449371e-05, -1.3552527156068805e-20, 2.5214795814616026e-21],
        [-1.3552527156068805e-20, 0.00035705300836722431, 4.4766275773390029e-22],
        [2.5214795814616026e-21, 4.4766275773390029e-22, 0.00043190422402162329]]

I_45 = [[0.00019232606045470671, 1.3552527156068805e-20, 1.2627017167609311e-20],
        [1.3552527156068805e-20, 0.00036422223881746503, 1.6650309163384415e-21],
        [1.2627017167609311e-20, 1.6650309163384415e-21, 0.0005466328545714452]]

I_60 = [[0.000311243867376448, 0, 0],
        [0, 0.000311243867376448, 0],
        [0, 0, 0.0006128367621210681]]

I_75 = [[0.00066921849216050014, 2.7105054312137611e-20, -3.3685125287159029e-21],
        [2.7105054312137611e-20, 0.00038309493110161302, 1.0621606046267175e-20],
        [-3.3685125287159029e-21, 1.0621606046267175e-20, 0.0010423041088969216]]

I_90 = [[0.0011706814995045239, 2.7105054312137611e-20, -3.3881317890172014e-21],
        [2.7105054312137611e-20, 0.00039647954261617555, 2.3523493101256951e-20],
        [-3.3881317890172014e-21, 2.3523493101256951e-20, 0.0015570834427368358]]

class postProcessing:

    def __init__(self, test_name):

        self.mass = {'30': 0.078742493606727915,'45': 0.081476344460819278,
                     '60': 0.060318578948924034, '75': 0.088985917618018107, 
                     '90': 0.094448719111790758}

        
        self.I = {'30': np.array(I_30), '45': np.array(I_45), 
                  '60': np.array(I_60), '75': np.array(I_75),
                  '90': np.array(I_90)}
        self.mu = 0.9

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

    def compute_contact_force(self):
        slope = self.surface_slope
        
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
        
        return  list(result.args)[0]
    
    def compute_slip_vel(self, com_vel: np.ndarray,
                         ang_vel: np.ndarray, r: np.ndarray):
        return com_vel + self.hat(r) @ ang_vel
    
    def compute_contact_force_error(self):
        self.compute_pos_mass()
        contact_forces = self.compute_contact_force()
        contact_forces = np.array(contact_forces).reshape(3,3)
        return contact_forces
    
    def compute_friction_force_error(self, contact_info,
                                     v, omega, pos):
        normal = contact_info[4:7]
        force = contact_info[7:10]
        
        ## contact force along normal (contact_force.normal)
        force_n = np.dot(force, normal)
        r_ac = contact_info[1:4] - pos
        slip_vel = self.compute_slip_vel(v, omega, r_ac)
        ## analytical solution
        friction_force_a  = self.mu * force_n * (slip_vel/np.linalg.norm(slip_vel))
        ## numerical solution
        friciton_force_n = contact_info[7:9]
        F_mag_error = np.linalg.norm(friciton_force_n[:2] - friction_force_a)
        F_dir_error = np.dot(v, friciton_force_n)

        return F_mag_error, F_dir_error
    
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
        self.F_mag_error = np.zeros((self.N,3))
        self.F_dir_error = np.zeros((self.N,3))
        self.contact_force_error = np.zeros((self.N, 3))
        
        for i in range(self.N):
            # angular velocity in body frame
            omega_w = omega[i].tolist()
            quat = rot[i].tolist()
            quat = Quaterniond(quat[0], quat[1], quat[2], quat[3])
            omega_b = quat.rotate_vector_reverse(Vector3d(omega_w[0], 
                                                 omega_w[1], omega_w[2]))
            omega_b = np.array([omega_b[0], omega_b[1], omega_b[2]])

            # translation energy + rotational energy + potential energy
            tran_E = 0.5*self.m*v[i].dot(v[i])
            rot_E = 0.5*omega_b.dot(self.I.dot(omega_b))
            V = - self.m*self.gravity.dot(pos[i])
            E[i] = tran_E + rot_E + V

            if self.complex:
                F_mag_error, F_dir_error = self.compute_friction_force_error(
                                                contact1_info[i], v[i], omega_b, 
                                                pos[i])
                self.F_mag_error[i, 0] = F_mag_error
                self.F_dir_error[i, 0] = F_dir_error
    
                F_mag_error, F_dir_error = self.compute_friction_force_error(
                                                contact2_info[i], v[i], omega_b, 
                                                pos[i])
                self.F_mag_error[i, 1] = F_mag_error
                self.F_dir_error[i, 1] = F_dir_error
    
                F_mag_error, F_dir_error = self.compute_friction_force_error(
                                                contact3_info[i], v[i], omega_b, 
                                                pos[i])
                self.F_mag_error[i, 2] = F_mag_error
                self.F_dir_error[i, 2] = F_dir_error
            else:
                contact_forces = self.compute_contact_force_error()
                self.contact_force_error[i, 0] = np.linalg.norm(contact_forces[0, :] - contact1_info[7:10])
                self.contact_force_error[i, 1] = np.linalg.norm(contact_forces[1, :] - contact2_info[7:10])
                self.contact_force_error[i, 2] = np.linalg.norm(contact_forces[2, :] - contact3_info[7:10])

        initial_energy = E[0]

        self.E_error = E - initial_energy
        

    def get_maxabs_error(self):
        self.E_error = self.E_error[self.E_error>0]
        self.E_maxabs_error = np.max(self.E_error)
        if self.complex:
            self.F_dir_error = self.F_dir_error[self.F_dir_error>0]
            self.F_maxabs_dir_error_a = np.max(self.F_dir_error[:, 0])
            self.F_maxabs_dir_error_b = np.max(self.F_dir_error[:, 1])
            self.F_maxabs_dir_error_c = np.max(self.F_dir_error[:, 2])
            self.F_maxabs_mag_error_a = np.max(self.F_mag_error[:, 0])
            self.F_maxabs_mag_error_b = np.max(self.F_mag_error[:, 1])
            self.F_maxabs_mag_error_c = np.max(self.F_mag_error[:, 2])
        else:
            self.contact_force_maxabs_error_a = np.max(self.contact_force_error[:, 0])
            self.contact_force_maxabs_error_a = np.max(self.contact_force_error[:, 1])
            self.contact_force_maxabs_error_a = np.max(self.contact_force_error[:, 2])



        
if __name__ == "__main__":
    dir = BENCHMARK_NAME
    print(f"BENCHMARK: {dir}")

    post_processing = postProcessing(dir)
    result_dir , file_names = post_processing.get_file_names(dir)
    file_names = sorted(file_names, reverse=True)

    for file in file_names:
        print(f"TEST: {file}")
        file_path = os.path.join(result_dir,file)
        config, states = post_processing.read_file(file_path)
        physic_engine = config[0,0]
        dt = config[0,1]
        complex = bool(config[0,2])
        collision = bool(config[0,3])
        modelCount = config[0,4]
        computation_time = config[0,5]
        log_multiple = bool(config[0,6])
        class_name = config[0,7]
            
        print(f" Physics engines: {physic_engine} \n Timestep: {dt} \n Complex: {complex} \n Number of models: {modelCount}")
        post_processing.set_test_parameters(physic_engine, dt, complex, collision, modelCount, computation_time, class_name)

        if log_multiple:
            no_of_models = modelCount
        else:
            no_of_models = 1 
        states_per_model = int(len(states[:,0])/no_of_models)
        states = states.reshape(no_of_models, states_per_model,-1)

        for i in range(no_of_models):
            print(f" => Model number: {i+1}")
            model_states = states[i]
            sim_time = model_states[:,0]

            post_processing.get_analytical_sol(sim_time, i)
            post_processing.cal_metrics(model_states)
            post_processing.save_metrics()
    
    post_processing.csv_file.close()
    print(f"final metrics path {post_processing.metrics_path}")
    data = pd.read_csv(post_processing.metrics_path)

    if dir=="BENCHMARK_boxes_dt":
        storted_data = data.sort_values(by='dt')
    else:
        storted_data = data.sort_values(by='modelCount')

    storted_data.to_csv(post_processing.metrics_path, index=False)



    




     
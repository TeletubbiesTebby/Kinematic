import sympy as sp

class Point:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

class Joint_Pos_Plot:
    def __init__(self, P1, P2, P3, PE):
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        self.PE = PE

class Joint:
    def __init__(self, q1=0, q2=0, q3=0):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

class RRR_Robot:
    def __init__(self, l1, l2, l3, q1, q2, q3):
        self.l1 = l1 
        self.l2 = l2
        self.l3 = l3
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

    def update_joint(self, q_joint):
        self.q1 = q_joint.q1
        self.q2 = q_joint.q2
        self.q3 = q_joint.q3

    def Inverse_Kinematics(self, goal_point):      # goal_point  Point(x,y,z)
        Px = goal_point.x
        Py = goal_point.y
        Pz = goal_point.z

        # Solution for q1
        q1_sol = [sp.atan2(Py, Px), sp.pi + sp.atan2(Py, Px)]

        # Solutions for q2
        r = sp.sqrt(Px**2 + Py**2 + (Pz - self.l1)**2)
        cos_row = (r**2 + self.l2**2 - self.l3**2) / (2 * r * self.l2)
        sin_row = [sp.sqrt(1 - cos_row**2), -sp.sqrt(1 - cos_row**2)]

        row = [sp.atan2(sin_row[0], cos_row), sp.atan2(sin_row[1], cos_row)]
        alpha = sp.atan2(Pz - self.l1, sp.sqrt(Px**2 + Py**2))

        q2_sol = [alpha - row[0], alpha - row[1]]

        # Solutions for q3
        cos_3 = (r**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)
        sin_3 = [sp.sqrt(1 - cos_3**2), -sp.sqrt(1 - cos_3**2)]

        q3_sol = [sp.atan2(sin_3[0], cos_3), sp.atan2(sin_3[1], cos_3)]

        goal_joint_space =  Joint(q1_sol[0], q2_sol[0], q3_sol[0])

        return goal_joint_space
    
    def Forward_Kinematics(self):

        P1 = Point(x=0, y=0, z=0)
        P2 = Point(x=0, y=0, z=self.l1)

        T0_1 = sp.Matrix([
            [sp.cos(self.q1), -sp.sin(self.q1), 0, 0],
            [sp.sin(self.q1), sp.cos(self.q1), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T1_2 = sp.Matrix([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, self.l1],
            [0, 0, 0, 1]
        ])

        T2_3 = sp.Matrix([
            [sp.cos(self.q2), -sp.sin(self.q2), 0, 0],
            [sp.sin(self.q2), sp.cos(self.q2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T3_4 = sp.Matrix([
            [sp.cos(self.q3), -sp.sin(self.q3), 0, self.l2],
            [sp.sin(self.q3), sp.cos(self.q3), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T4_E = sp.Matrix([
            [1, 0, 0, self.l3],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        T0_4 = sp.simplify(T0_1 * T1_2 * T2_3 * T3_4)
        P3 = Point(x=T0_4[0, 3], y=T0_4[1, 3], z=T0_4[2, 3])
        # print("P3: ", P3.x, " , ", P3.y, " , ", P3.z)

        # Compute the overall transformation matrix T0_E
        T0_E = sp.simplify(T0_4 * T4_E)
        PE = Point(x=T0_E[0, 3], y=T0_E[1, 3], z=T0_E[2, 3])
        # print("PE: ", PE.x, " , ", PE.y, " , ", PE.z)

        joint_pos_plot = Joint_Pos_Plot(P1, P2, P3, PE)

        return  joint_pos_plot # (P1, P2, P3, PE) #นิ้งจีงต้องใช้


def Robot_plot(joint_pos):
    p1 = joint_pos.P1
    p2 = joint_pos.P2
    p3 = joint_pos.P3
    pE = joint_pos.PE





RRR = RRR_Robot(l1=2, l2=1, l3=0.4, q1=0.2, q2=1.3, q3=0.8)

# check Forward_Kinematics 
goal_point = RRR.Forward_Kinematics()
print("goal point")
print("Px ", goal_point.PE.x)
print("Py ", goal_point.PE.y)
print("Pz ", goal_point.PE.z)

#check Inverse_Kinematic
goal_joint_space = RRR.Inverse_Kinematics(goal_point.PE)
print("goal joint space")
print("q1_sol ", goal_joint_space.q1)
print("q2_sol ", goal_joint_space.q2)
print("q3_sol ", goal_joint_space.q3)

RRR.update_joint(goal_joint_space)

position = RRR.Forward_Kinematics()
print("goal point")
print("Px ", position.PE.x)
print("Py ", position.PE.y)
print("Pz ", position.PE.z)

# PLot check
Robot_plot(goal_point)

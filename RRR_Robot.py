import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from math import pi
import numpy as np

# Simulation parameters
M = 100

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

        return  joint_pos_plot # (P1, P2, P3, PE)


# ฟังก์ชันสุ่มสร้างสิ่งกีดขวาง
def generate_random_obstacles(num_obstacles=3, max_radius=0.5, space_limit=2, min_distance=0.1):
    """
    สร้างสิ่งกีดขวางแบบสุ่มในพื้นที่ 3D โดยไม่มีการทับซ้อน
    :param num_obstacles: จำนวนสิ่งกีดขวางที่ต้องการ
    :param max_radius: รัศมีสูงสุดของสิ่งกีดขวาง
    :param space_limit: ขอบเขตของพื้นที่ (x, y, z จะถูกจำกัดใน -space_limit ถึง +space_limit)
    :param min_distance: ระยะขั้นต่ำระหว่างจุดศูนย์กลางของสิ่งกีดขวาง (ไม่รวมผลรวมรัศมี)
    :return: รายการสิ่งกีดขวางในรูปแบบ [[x, y, z, r], ...]
    """
    obstacles = []

    for _ in range(num_obstacles):
        while True:  # วนลูปจนกว่าจะหาตำแหน่งที่ไม่ชนได้
            x = random.uniform(-space_limit, space_limit)
            y = random.uniform(-space_limit, space_limit)
            z = random.uniform(-space_limit, space_limit)
            r = random.uniform(0.1, max_radius)  # รัศมีขั้นต่ำ 0.1

            # ตรวจสอบว่ามีการชนกับสิ่งกีดขวางที่สร้างไปแล้วหรือไม่
            valid = True
            for existing_obstacle in obstacles:
                ex, ey, ez, er = existing_obstacle
                distance = np.sqrt((x - ex)**2 + (y - ey)**2 + (z - ez)**2)
                if distance < r + er + min_distance:  # ระยะทางต้องมากกว่ารัศมีรวมและระยะขั้นต่ำ
                    valid = False
                    break

            if valid:
                obstacles.append([x, y, z, r])
                break

    return obstacles

def detect_collision(obstacles, joint_position):
    """
    ตรวจสอบว่าตำแหน่งของข้อต่อ (joint_position) อยู่ใกล้กับสิ่งกีดขวาง (obstacles) หรือไม่
    """
    for obstacle in obstacles:
        obstacle_center = np.array(obstacle[:3])  # จุดศูนย์กลางของสิ่งกีดขวาง
        radius = obstacle[3]                     # รัศมีของสิ่งกีดขวาง
        joint_vec = np.array(joint_position)     # ตำแหน่งของข้อต่อ

        # คำนวณระยะทางระหว่างข้อต่อและจุดศูนย์กลางของสิ่งกีดขวาง
        distance = np.linalg.norm(joint_vec - obstacle_center)

        # ตรวจสอบว่าระยะทางน้อยกว่าหรือเท่ากับรัศมีหรือไม่
        if distance <= radius:
            return 1  # มีการชนหรือใกล้สิ่งกีดขวาง
    return 0  # ไม่มีการชน

def get_occupancy_grid(arm, obstacles):
    grid = np.zeros((M, M, M), dtype=int)  # สร้าง Grid 3 มิติ
    theta_list = [2 * i * pi / M for i in range(M)]  # มุมในเรเดียน

    for i in range(M):
        for j in range(M):
            for k in range(M):
                arm.update_joints([theta_list[i], theta_list[j], theta_list[k]])
                points = arm.points  # ได้ตำแหน่งของแขนกลแต่ละจุด
                collision_detected = False

                # ตรวจสอบการชนในแต่ละลิงก์
                for point in points:
                    if detect_collision(obstacles, point):
                        collision_detected = True
                        break

                # อัปเดต Grid
                grid[i][j][k] = int(collision_detected)
    return grid


# PLot check
def Show_plot(joint_pos, obstacles):
    # Extract joint positions
    p1 = [float(joint_pos.P1.x), float(joint_pos.P1.y), float(joint_pos.P1.z)]
    p2 = [float(joint_pos.P2.x), float(joint_pos.P2.y), float(joint_pos.P2.z)]
    p3 = [float(joint_pos.P3.x), float(joint_pos.P3.y), float(joint_pos.P3.z)]
    pE = [float(joint_pos.PE.x), float(joint_pos.PE.y), float(joint_pos.PE.z)]

    # Extract coordinates for links
    x_coords = [p1[0], p2[0], p3[0], pE[0]]
    y_coords = [p1[1], p2[1], p3[1], pE[1]]
    z_coords = [p1[2], p2[2], p3[2], pE[2]]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the joints
    ax.scatter(*p1, color='red', s=100, label='Base (P1)')
    ax.scatter(*p2, color='blue', s=100, label='Joint 1 (P2)')
    ax.scatter(*p3, color='green', s=100, label='Joint 2 (P3)')
    ax.scatter(*pE, color='purple', s=100, label='End Effector (PE)')

    # Plot the links
    ax.plot(x_coords, y_coords, z_coords, color='black', label='Robot Links')

    # วาดสิ่งกีดขวาง (ทรงกลม 3 มิติ)
    for obstacle in obstacles:
        u = np.linspace(0, 2 * pi, 100)  # มุมสำหรับสร้างทรงกลม
        v = np.linspace(0, pi, 100)
        x = obstacle[3] * np.outer(np.cos(u), np.sin(v)) + obstacle[0]
        y = obstacle[3] * np.outer(np.sin(u), np.sin(v)) + obstacle[1]
        z = obstacle[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle[2]
        ax.plot_surface(x, y, z, color='r', edgecolor='none')  # สีฟ้าพร้อมค่าความโปร่งใส

    # Add labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
    # ax.text(joint_pos.PE.x, joint_pos.PE.y, joint_pos.PE.z, f"({joint_pos.PE.x:.2f}, {joint_pos.PE.y:.2f}, {joint_pos.PE.z:.2f})", color='orange')

    # Set axis limits
    max_range = float(max(
        max(abs(coord) for coord in x_coords),
        max(abs(coord) for coord in y_coords),
        max(abs(coord) for coord in z_coords)
    ))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])

    # Show the plot
    plt.show()

def main():
    RRR = RRR_Robot(l1=1, l2=1, l3=0.4, q1=0.1, q2=-0.5, q3=0.8)

    # สร้างสิ่งกีดขวางแบบสุ่ม
    obstacles = generate_random_obstacles()

    print(obstacles)

    # check Forward_Kinematics 
    goal_point = RRR.Forward_Kinematics()
    print("goal point")
    print("Px ", goal_point.PE.x)
    print("Py ", goal_point.PE.y)
    print("Pz ", goal_point.PE.z)

    # check Inverse_Kinematic
    # goal_joint_space = RRR.Inverse_Kinematics(goal_point.PE)
    # print("goal joint space")
    # print("q1_sol ", goal_joint_space.q1)
    # print("q2_sol ", goal_joint_space.q2)
    # print("q3_sol ", goal_joint_space.q3)

    # RRR.update_joint(goal_joint_space)

    # position = RRR.Forward_Kinematics()
    # print("goal point")
    # print("Px ", position.PE.x)
    # print("Py ", position.PE.y)
    # print("Pz ", position.PE.z)

    # Plot the robot
    Show_plot(goal_point, obstacles)

if __name__ == '__main__':
    main()
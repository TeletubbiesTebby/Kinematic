from math import pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.ion()

# Simulation parameters
M = 100
obstacles = [
    [1.75, 0.75, 0.6, 0.3],  # ทรงกลมที่จุด (1.75, 0.75, 0.6) รัศมี 0.3
    [0.55, 1.5, 1.2, 0.5],   # ทรงกลมที่จุด (0.55, 1.5, 1.2) รัศมี 0.5
    [0, -1, 0.5, 0.25]       # ทรงกลมที่จุด (0, -1, 0.5) รัศมี 0.25
]

def main():
    arm = NLinkArm([1, 1, 1], [0, 0, 0])  # แขนกล 3 ข้อต่อ
    start = (10, 50, 20)  # ตำแหน่งเริ่มต้นใน Grid
    goal = (58, 56, 45)   # ตำแหน่งเป้าหมายใน Grid

    grid = get_occupancy_grid(arm, obstacles)
    plt.imshow(grid.max(axis=2))  # แสดงภาพรวม Grid (มุมมอง 2D)
    plt.show()

    route = astar_torus(grid, start, goal)
    
    # วาดเส้นทางและแสดงการเคลื่อนไหว
    for node in route:
        theta1 = 2 * pi * node[0] / M - pi
        theta2 = 2 * pi * node[1] / M - pi
        theta3 = 2 * pi * node[2] / M - pi
        arm.update_joints([theta1, theta2, theta3])
        arm.plot(obstacles=obstacles)

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

def astar_torus(grid, start_node, goal_node):
    grid[start_node] = 4
    grid[goal_node] = 5
    parent_map = np.full((M, M, M), None, dtype=object)
    heuristic_map = calc_heuristic_map_3d(M, goal_node)

    explored_heuristic_map = np.full((M, M, M), np.inf)
    distance_map = np.full((M, M, M), np.inf)
    explored_heuristic_map[start_node] = heuristic_map[start_node]
    distance_map[start_node] = 0

    while True:
        current_node = np.unravel_index(np.argmin(explored_heuristic_map, axis=None), explored_heuristic_map.shape)
        if current_node == goal_node or np.isinf(explored_heuristic_map[current_node]):
            break

        explored_heuristic_map[current_node] = np.inf
        for neighbor in find_neighbors(*current_node):
            if grid[neighbor] == 0 or grid[neighbor] == 5:
                tentative_distance = distance_map[current_node] + 1
                if tentative_distance < distance_map[neighbor]:
                    distance_map[neighbor] = tentative_distance
                    explored_heuristic_map[neighbor] = distance_map[neighbor] + heuristic_map[neighbor]
                    parent_map[neighbor] = current_node

    route = []
    if np.isinf(distance_map[goal_node]):
        print("No route found.")
    else:
        current = goal_node
        while current:
            route.insert(0, current)
            current = parent_map[current]
        print("Route found with %d steps." % len(route))
    return route

def find_neighbors(i, j, k):
    neighbors = []
    for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        ni, nj, nk = (i + di) % M, (j + dj) % M, (k + dk) % M
        neighbors.append((ni, nj, nk))
    return neighbors

def calc_heuristic_map_3d(M, goal_node):
    X, Y, Z = np.meshgrid(range(M), range(M), range(M), indexing="ij")
    gx, gy, gz = goal_node
    heuristic_map = np.minimum(
        np.abs(X - gx), M - np.abs(X - gx)) + \
        np.minimum(np.abs(Y - gy), M - np.abs(Y - gy)) + \
        np.minimum(np.abs(Z - gz), M - np.abs(Z - gz))
    return heuristic_map

class NLinkArm:
    def __init__(self, link_lengths, joint_angles):
        self.n_links = len(link_lengths)
        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0, 0] for _ in range(self.n_links + 1)]
        self.lim = sum(link_lengths)
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        self.points[0] = [0, 0, 0]  # จุดเริ่มต้น
        for i in range(1, self.n_links + 1):
            x = self.points[i - 1][0] + self.link_lengths[i - 1] * np.cos(self.joint_angles[i - 1])
            y = self.points[i - 1][1] + self.link_lengths[i - 1] * np.sin(self.joint_angles[i - 1])
            z = self.points[i - 1][2]  # ไม่มีการเปลี่ยนในแกน z (ถ้าต้องการแก้เพิ่มในอนาคต)
            self.points[i] = [x, y, z]

    def plot(self, obstacles=[]):
        # ตรวจสอบว่ากราฟมีแกน 3D หรือไม่ ถ้ายังไม่มีให้สร้างใหม่
        fig = plt.gcf()  # ใช้กราฟที่มีอยู่
        if not hasattr(fig, '_ax3d'):  # ถ้ายังไม่มีแกน 3D
            ax = fig.add_subplot(111, projection='3d')  # สร้างแกน 3D
            fig._ax3d = ax  # บันทึกแกน 3D ไว้ในตัวแปรของกราฟ
        else:
            ax = fig._ax3d  # ใช้แกน 3D ที่มีอยู่

        ax.cla()  # ล้างข้อมูลเก่าในกราฟ

        # วาดสิ่งกีดขวาง (ทรงกลม 3 มิติ)
        for obstacle in obstacles:
            u = np.linspace(0, 2 * pi, 100)  # มุมสำหรับสร้างทรงกลม
            v = np.linspace(0, pi, 100)
            x = obstacle[3] * np.outer(np.cos(u), np.sin(v)) + obstacle[0]
            y = obstacle[3] * np.outer(np.sin(u), np.sin(v)) + obstacle[1]
            z = obstacle[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle[2]
            ax.plot_surface(x, y, z, color='r', edgecolor='none')  # สีฟ้าพร้อมค่าความโปร่งใส

        # วาดแขนกล
        for i in range(len(self.points) - 1):
            x = [self.points[i][0], self.points[i + 1][0]]
            y = [self.points[i][1], self.points[i + 1][1]]
            z = [self.points[i][2], self.points[i + 1][2]]
            ax.plot(x, y, z, 'bo-')

        # ตั้งค่าขอบเขต
        ax.set_xlim([-self.lim, self.lim])
        ax.set_ylim([-self.lim, self.lim])
        ax.set_zlim([-self.lim, self.lim])
        plt.draw()
        plt.pause(0.1)  # พักเล็กน้อยเพื่ออัปเดตการแสดงผล


if __name__ == '__main__':
    main()

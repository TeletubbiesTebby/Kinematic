from math import pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
plt.ion()

# Simulation parameters
M = 100

def main():
    arm = NLinkArm([1, 1, 1], [0, 0, 0])  # แขนกล 3 ข้อต่อ
    start = (10, 50, 20)  # ตำแหน่งเริ่มต้นใน Grid
    goal = (58, 56, 45)   # ตำแหน่งเป้าหมายใน Grid

    # สร้างสิ่งกีดขวางแบบสุ่ม
    obstacles = generate_random_obstacles()

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

        # ตรวจสอบการชนและแสดงสถานะพร้อมระยะห่าง
        collision_detected = False
        print("\nChecking for collisions...")
        for point_index, point in enumerate(arm.points):
            for obstacle_index, obstacle in enumerate(obstacles):
                obstacle_center = np.array(obstacle[:3])
                radius = obstacle[3]
                distance = np.linalg.norm(np.array(point) - obstacle_center)
                if distance <= radius:  # มีการชน
                    collision_detected = True
                    print(f"Collision detected at joint {point_index + 1} with obstacle {obstacle_index + 1}:")
                    print(f"  Joint Position: {point}, Obstacle Center: {obstacle_center}, Radius: {radius:.2f}, Distance: {distance:.2f}")
                else:
                    print(f"No collision at joint {point_index + 1} with obstacle {obstacle_index + 1}:")
                    print(f"  Joint Position: {point}, Obstacle Center: {obstacle_center}, Radius: {radius:.2f}, Distance: {distance:.2f}")

        if not collision_detected:
            print("All joints are safe.")

        # แสดงกราฟของแขนกล
        arm.plot(obstacles=obstacles)



# ฟังก์ชันสุ่มสร้างสิ่งกีดขวาง
def generate_random_obstacles(num_obstacles=3, max_radius=0.8, space_limit=3, min_distance=0.1):
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

def detect_collision(obstacles, joint_position, buffer=0.2):
    """
    ตรวจสอบว่าตำแหน่งของข้อต่อ (joint_position) อยู่ใกล้กับสิ่งกีดขวาง (obstacles) หรือไม่
    โดยเพิ่มระยะเผื่อเฉียด (buffer)
    :param obstacles: รายการสิ่งกีดขวางในรูปแบบ [[x, y, z, r], ...]
    :param joint_position: ตำแหน่งของข้อต่อในรูปแบบ [x, y, z]
    :param buffer: ระยะเผื่อเฉียดที่เพิ่มจากรัศมีของสิ่งกีดขวาง
    :return: 1 ถ้ามีการชนหรืออยู่ในระยะเฉียด, 0 มิฉะนั้น
    """
    for obstacle in obstacles:
        obstacle_center = np.array(obstacle[:3])  # จุดศูนย์กลางของสิ่งกีดขวาง
        radius = obstacle[3]                     # รัศมีของสิ่งกีดขวาง
        joint_vec = np.array(joint_position)     # ตำแหน่งของข้อต่อ

        # คำนวณระยะทางระหว่างข้อต่อและจุดศูนย์กลางของสิ่งกีดขวาง
        distance = np.linalg.norm(joint_vec - obstacle_center)

        # ตรวจสอบว่าระยะทางน้อยกว่าหรือเท่ากับ (รัศมี + buffer) หรือไม่
        if distance <= radius + buffer:
            return 1  # มีการชนหรืออยู่ในระยะเฉียด
    return 0  # ไม่มีการชนหรือเฉียด


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
            ax.plot_surface(x, y, z, color='r', edgecolor='none')  

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

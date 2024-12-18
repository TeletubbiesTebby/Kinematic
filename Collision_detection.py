from math import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors

plt.ion()

# Simulation parameters
M = 100
obstacles = [[1.75, 0.75, 0.6], [0.55, 1.5, 0.5], [0, -1, 0.25]]

def main():
    arm = NLinkArm([1, 1, 1], [0, 0, 0])  # 3 ข้อต่อ
    start = (10, 50, 20)
    goal = (58, 56, 45)
    grid = get_occupancy_grid(arm, obstacles)
    plt.imshow(grid.max(axis=2))
    plt.show()
    # wait_for_esc() 

    route = astar_torus(grid, start, goal)
    for node in route:
        theta1 = 2 * pi * node[0] / M - pi
        theta2 = 2 * pi * node[1] / M - pi
        theta3 = 2 * pi * node[2] / M - pi
        arm.update_joints([theta1, theta2, theta3])
        arm.plot(obstacles=obstacles)

def detect_collision(line_seg, circle):
    a_vec = np.array([line_seg[0][0], line_seg[0][1]])
    b_vec = np.array([line_seg[1][0], line_seg[1][1]])
    c_vec = np.array([circle[0], circle[1]])
    radius = circle[2]
    line_vec = b_vec - a_vec
    line_mag = np.linalg.norm(line_vec)
    circle_vec = c_vec - a_vec
    proj = circle_vec.dot(line_vec / line_mag)
    if proj <= 0:
        closest_point = a_vec
    elif proj >= line_mag:
        closest_point = b_vec
    else:
        closest_point = a_vec + line_vec * proj / line_mag
    return np.linalg.norm(closest_point - c_vec) <= radius

def get_occupancy_grid(arm, obstacles):
    grid = np.zeros((M, M, M), dtype=int)
    theta_list = [2 * i * pi / M for i in range(-M // 2, M // 2 + 1)]
    for i in range(M):
        for j in range(M):
            for k in range(M):
                arm.update_joints([theta_list[i], theta_list[j], theta_list[k]])
                points = arm.points
                collision_detected = False
                for link_idx in range(len(points) - 1):
                    for obstacle in obstacles:
                        line_seg = [points[link_idx], points[link_idx + 1]]
                        collision_detected = detect_collision(line_seg, obstacle)
                        if collision_detected:
                            break
                    if collision_detected:
                        break
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

class NLinkArm(object):
    def __init__(self, link_lengths, joint_angles):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()
        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]
        self.lim = sum(link_lengths)
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * np.sin(np.sum(self.joint_angles[:i]))
        self.end_effector = np.array(self.points[self.n_links]).T

    def plot(self, obstacles=[]):
        plt.cla()
        for obstacle in obstacles:
            circle = plt.Circle((obstacle[0], obstacle[1]), radius=0.5 * obstacle[2], fc='k')
            plt.gca().add_patch(circle)
        for i in range(self.n_links + 1):
            if i is not self.n_links:
                plt.plot([self.points[i][0], self.points[i + 1][0]],
                         [self.points[i][1], self.points[i + 1][1]], 'r-')
            plt.plot(self.points[i][0], self.points[i][1], 'k.')
        plt.xlim([-self.lim, self.lim])
        plt.ylim([-self.lim, self.lim])
        plt.draw()
        plt.pause(1e-5)

if __name__ == '__main__':
    main()

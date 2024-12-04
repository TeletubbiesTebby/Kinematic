import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from math import pi
import numpy as np
import heapq
import math
from matplotlib.animation import FuncAnimation
from itertools import permutations


#---------------------------------Simulation parameters----------------------------------#
M = 100

obstacles = [
    [1.5, -2, 3, 0.7],#x,y,z รัศมี 
    [2.5, 2, 4, 0.5], 
    [-2.5, 0, 1.2, 0.5],   
    [0, -1, 3, 0.5],      
    [-1.75, 2.5 , 0.5, 0.5 , 0.8]       
]

#-------------------------------------Define Class----------------------------------------#

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
    
    def get_angles(self):
        return [self.q1, self.q2, self.q3]
    



class P2P_Path:
    def __init__(self, i, j, start, path, total_dist):
        self.i = i
        self.j = j
        self.start = start
        self.path = path
        self.total_dist = total_dist
    
#---------------------------Toroidal Grid & Obstacle detection-----------------------------#

def generate_random_obstacles(num_obstacles=3, max_radius=0.5, x_limit=(-3, 3), y_limit=(-3, 3), z_limit=(0, 6), min_distance=0.1):
    """
    Generate random obstacles in the 3D workspace without overlapping.
    Obstacles are constrained within the given limits.
    :param num_obstacles: Number of obstacles to generate.
    :param max_radius: Maximum radius of the obstacles.
    :param x_limit: X-axis limits for obstacles (tuple of min and max).
    :param y_limit: Y-axis limits for obstacles (tuple of min and max).
    :param z_limit: Z-axis limits for obstacles (tuple of min and max).
    :param min_distance: Minimum distance between obstacles.
    :return: List of obstacles in the form [[x, y, z, r], ...]
    """
    obstacles = []

    for _ in range(num_obstacles):
        while True:
            x = random.uniform(x_limit[0], x_limit[1])
            y = random.uniform(y_limit[0], y_limit[1])
            z = random.uniform(z_limit[0], z_limit[1])
            r = random.uniform(0.1, max_radius)

            # Ensure the obstacle does not overlap with existing ones
            valid = True
            for existing_obstacle in obstacles:
                ex, ey, ez, er = existing_obstacle
                distance = np.sqrt((x - ex) ** 2 + (y - ey) ** 2 + (z - ez) ** 2)
                if distance < r + er + min_distance:
                    valid = False
                    break

            if valid:
                obstacles.append([x, y, z, r])
                break

    return obstacles

def detect_collision(obstacles, joint_positions):
    """
    Check if the joint positions or the arms (segments between joints) are close to any obstacles.
    :param obstacles: List of obstacles in the form [[x, y, z, r], ...].
    :param joint_positions: 2D list of joint positions [[x1, y1, z1], [x2, y2, z2], ...].
    :return: 1 if there is a collision, 0 otherwise.
    """
    def point_to_line_distance(point, line_start, line_end):
        """
        Calculate the shortest distance from a point to a line segment.
        :param point: The point [x, y, z].
        :param line_start: Start point of the line segment [x, y, z].
        :param line_end: End point of the line segment [x, y, z].
        :return: The shortest distance.
        """
        # Convert all inputs to float arrays to ensure compatibility with numpy
        line_start = np.array([float(coord.evalf()) if hasattr(coord, "evalf") else float(coord) for coord in line_start], dtype=float)
        line_end = np.array([float(coord.evalf()) if hasattr(coord, "evalf") else float(coord) for coord in line_end], dtype=float)
        point_obstacle = np.array([float(coord) for coord in point], dtype=float)

        line_vec = line_end - line_start
        point_vec = point_obstacle - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len > 0:
            line_unit_vec = line_vec / line_len
        else:
            line_unit_vec = line_vec  # Avoid division by zero for degenerate segments
        projection = np.dot(point_vec, line_unit_vec)

        if projection < 0:
            # Closest point is the start of the line
            closest_point = line_start
        elif projection > line_len:
            # Closest point is the end of the line
            closest_point = line_end
        else:
            # Closest point is somewhere along the line
            closest_point = line_start + projection * line_unit_vec

        return np.linalg.norm(point - closest_point)

    # Initialize variables to track distances when no collision occurs
    min_joint_distance = float('inf')
    min_segment_distance = float('inf')

    # Check collision for each joint
    for joint_position in joint_positions:
        joint_vec = np.array([float(coord.evalf()) if hasattr(coord, "evalf") else float(coord) for coord in joint_position])  # Convert to float

        for obstacle in obstacles:
            obstacle_center = np.array([float(coord) for coord in obstacle[:3]])  # Convert to float
            radius = float(obstacle[3])  # Convert to float

            # Calculate distance
            distance = np.linalg.norm(joint_vec - obstacle_center)

            # Track the smallest distance if no collision
            if distance < min_joint_distance:
                min_joint_distance = distance

            # Check if within the radius
            if distance <= radius:
                print(f"⚠️  Collision detected at joint position {joint_vec}\nwith obstacle {obstacle_center}.")
                return 1  # Collision detected with a joint

    # Check collision for each arm segment
    for i in range(len(joint_positions) - 1):
        line_start = joint_positions[i]
        line_end = joint_positions[i + 1]

        for obstacle in obstacles:
            obstacle_center = np.array([float(coord) for coord in obstacle[:3]])  # Convert to float
            radius = float(obstacle[3])  # Convert to float

            # Calculate the shortest distance from the obstacle to the arm segment
            distance = point_to_line_distance(obstacle_center, line_start, line_end)

            # Track the smallest distance if no collision
            if distance < min_segment_distance:
                min_segment_distance = distance

            if distance <= radius:
                print(f"⚠️  Collision detected with arm segment between {list(map(float, line_start))}\nand {list(map(float, line_end))}\n"
                      f"and obstacle {obstacle_center}.")
                return 1  # Collision detected with an arm segment

    # Print minimum distances if no collision occurs
    print(f"✅ No collision detected.")
    print(f"Minimum distance between joints and obstacles: {min_joint_distance:.2f}")
    print(f"Minimum distance between arm segments and obstacles: {min_segment_distance:.2f}")
    return 0  # No collision




def get_occupancy_grid(arm, obstacles):
    grid = np.zeros((M, M, M), dtype=int)  # Create a 3D grid
    theta_list = [2 * i * pi / M for i in range(M)]  # Generate joint angles

    for i in range(M):
        for j in range(M):
            for k in range(M):
                # Update robot's joint angles
                new_joint = Joint(theta_list[i], theta_list[j], theta_list[k])
                arm.update_joint(new_joint)

                # Get joint positions
                joint_points = arm.Forward_Kinematics()  # Get forward kinematics results
                points = convert_goal_point_to_2d_list(joint_points)  # Convert to 2D list

                # Check for collisions
                collision_detected = detect_collision(obstacles, points)

                # Update grid based on collision detection
                grid[i][j][k] = collision_detected
    return grid


def convert_goal_point_to_2d_list(goal_point):
    """
    Convert a Joint_Pos_Plot object containing Point objects into a 2D list.
    :param goal_point: Joint_Pos_Plot object (contains P1, P2, P3, PE)
    :return: 2D list with coordinates of P1, P2, P3, PE
    """
    return [
        [goal_point.P1.x, goal_point.P1.y, goal_point.P1.z],
        [goal_point.P2.x, goal_point.P2.y, goal_point.P2.z],
        [goal_point.P3.x, goal_point.P3.y, goal_point.P3.z],
        [goal_point.PE.x, goal_point.PE.y, goal_point.PE.z]
    ]

# Function for saving toroidal grid
def save_tor_grid(tor_grid, filename):
    np.save(filename, tor_grid)

def load_tor_grid(filename):
    return np.load(filename, allow_pickle=True)

def angle_to_grid_index(theta, grid_size):
    """Maps a joint angle to its corresponding grid index."""
    # theta = (theta + pi) % (2 * pi)  # Normalize angle to [0, 2pi]
    if theta < 0:
        theta += (2*pi)
    index = int((theta / (2 * pi)) * grid_size)
    if index == 100:
        index = 0
    return index

#---------------------------------------RRR Robot------------------------------------------#

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
    
#---------------------------------------A* Algorithm---------------------------------------#

# Define 26 possible moves (including diagonals in all 3 dimensions)
MOVEMENTS = [
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),  # 6 face neighbors
    (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),  # Diagonal neighbors on x-y plane
    (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),  # Diagonal neighbors on x-z plane
    (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),  # Diagonal neighbors on y-z plane
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),  # Full diagonal neighbors
    (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)  # Full diagonal neighbors
]

def heuristic(a, b, M):
    """Euclidean distance heuristic with toroidal wrap-around."""
    dx = min(abs(a[0] - b[0]), M - abs(a[0] - b[0]))  # Wrap around on x-axis
    dy = min(abs(a[1] - b[1]), M - abs(a[1] - b[1]))  # Wrap around on y-axis
    dz = min(abs(a[2] - b[2]), M - abs(a[2] - b[2]))  # Wrap around on z-axis
    return math.sqrt(dx**2 + dy**2 + dz**2)

def reconstruct_path(came_from, current):
    """Reconstruct the path from the goal to the start."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def toroidal_wrap(x, grid_size):
    """Handle toroidal wrap-around for 3D grid."""
    return (x + grid_size) % grid_size

def map_joint_to_grid_indices(start_joint, goal_joint, M):
    # Map each joint angle to the corresponding grid index for start_joint
    start_indices = (angle_to_grid_index(start_joint.q1, M), 
                     angle_to_grid_index(start_joint.q2, M), 
                     angle_to_grid_index(start_joint.q3, M))
    
    # Map each joint angle to the corresponding grid index for goal_joint
    goal_indices = (angle_to_grid_index(goal_joint.q1, M), 
                    angle_to_grid_index(goal_joint.q2, M), 
                    angle_to_grid_index(goal_joint.q3, M))
    
    # Output the joint angles and corresponding grid indices
    print("Start joint angles (q1, q2, q3):", start_joint.q1, ",", start_joint.q2, ",", start_joint.q3)
    print("Start indices in grid:", start_indices)
    
    print("Goal joint angles (q1, q2, q3):", goal_joint.q1, ",", goal_joint.q2, ",", goal_joint.q3)
    print("Goal indices in grid:", goal_indices)
    
    return start_indices, goal_indices

def astar_torus(grid, start, goal, M):
    """
    A* algorithm for pathfinding in a toroidal grid.
    
    Parameters:
    - grid: The occupancy grid (M x M x M).
    - start: The start position in grid indices (x, y, z).
    - goal: The goal position in grid indices (x, y, z).
    - M: The size of the grid (M x M x M).
    
    Returns:
    - The path from start to goal as a list of positions, or None if no path is found.
    """
    # Priority queue for A* (min-heap)
    open_list = []
    heapq.heappush(open_list, (0, start))  # (cost, position)
    
    # Dictionaries to store g, f, and came_from
    g_costs = {start: 0}  # Cost to reach each node
    f_costs = {start: heuristic(start, goal, M)}  # f = g + h
    came_from = {}  # To reconstruct the path
    
    while open_list:
        # Pop the node with the lowest f value
        _, current = heapq.heappop(open_list)
        
        # If we reached the goal, reconstruct the path
        if current == goal:
            path = reconstruct_path(came_from, current)
            return path
        
        # Explore neighbors
        for dx, dy, dz in MOVEMENTS:
            neighbor = (toroidal_wrap(current[0] + dx, M),
                       toroidal_wrap(current[1] + dy, M),
                       toroidal_wrap(current[2] + dz, M))
            
            if grid[neighbor[0]][neighbor[1]][neighbor[2]] == 1:  # Skip obstacles
                continue
            
            tentative_g_cost = g_costs[current] + 1  # Each move costs 1
            
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                came_from[neighbor] = current
                g_costs[neighbor] = tentative_g_cost
                f_costs[neighbor] = tentative_g_cost + heuristic(neighbor, goal, M)
                heapq.heappush(open_list, (f_costs[neighbor], neighbor))
    
    return None  # No path found

# Function to visualize the 3D A* path
def plot_path(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x_vals = [p[0] for p in path]
    y_vals = [p[1] for p in path]
    z_vals = [p[2] for p in path]
    
    ax.plot(x_vals, y_vals, z_vals, marker='o')
    plt.show()

#---------------------------------Constraint Detection--------------------------------#

# ตรวจสอบว่า RRR Robot สามารถ Reach the goal ได้มั๊ย
def is_reachable(goal_joint):
    """
    Checks if the inverse kinematics solution is valid.
    - If None is returned, the goal is unreachable.
    - If joint angles contain complex components, the goal is unreachable.
    """
    # Check if the goal_joint is None, meaning no solution was found
    if goal_joint is None:
        print("The goal is unreachable. No valid solution found.")
        return False
    
    # Check if any of the joint angles are complex numbers (using sympy to check for complex components)
    if isinstance(goal_joint.q1, sp.Basic) and not goal_joint.q1.is_real:
        print("The goal is unreachable. Complex joint angle q1 found.")
        return False
    if isinstance(goal_joint.q2, sp.Basic) and not goal_joint.q2.is_real:
        print("The goal is unreachable. Complex joint angle q2 found.")
        return False
    if isinstance(goal_joint.q3, sp.Basic) and not goal_joint.q3.is_real:
        print("The goal is unreachable. Complex joint angle q3 found.")
        return False

    # If no complex components were found
    print("The goal is reachable. Real joint angles found.")
    return True

#---------------------------------------TSP------------------------------------------#

def calculate_joint_distance(path, M):
    """
    Calculates the total angular distance that each joint has to move along the path.
    Assumes path is a list of tuples representing joint angles in grid indices.
    
    Parameters:
    - path: A list of tuples representing joint angles at each step in the path (in grid indices).
    - M: The size of the grid (number of discrete steps per joint angle).
    
    Returns:
    - q1_dist, q2_dist, q3_dist: The total distance each joint moves along the path (in radians).
    """
    def angle_distance(a1, a2, M):
        """Calculate the shortest angular distance between two grid indices."""
        # Convert grid indices to angles
        scale = (2 * pi) / M
        theta1 = a1 * scale
        theta2 = a2 * scale
        
        # Calculate the shortest distance (accounting for the toroidal wraparound)
        delta_theta1 = abs(theta2 - theta1)
        delta_theta2 = (2*pi) - abs(theta2 - theta1)

        if(delta_theta1 < delta_theta2):
            delta_theta = delta_theta1
        else:
            delta_theta = delta_theta2
        
        # delta_theta = (theta2 - theta1 + pi) % (2 * pi) - pi
        return delta_theta
    
    # Initialize the total distances for each joint
    q1_dist = 0
    q2_dist = 0
    q3_dist = 0
    
    # Calculate the total angular distance for each joint
    for i in range(1, len(path)):
        q1_dist += angle_distance(path[i-1][0], path[i][0], M)
        q2_dist += angle_distance(path[i-1][1], path[i][1], M)
        q3_dist += angle_distance(path[i-1][2], path[i][2], M)
    
    return q1_dist, q2_dist, q3_dist

# def TSP(RRR, start_joint, goal_points):
#     # เก็บระยะทางทั้งหมดที่ใช้ในการคำนวณการหมุน joint
#     best_sequence = None
#     min_joint_change = float('inf')  # เริ่มต้นด้วยค่ามากสุด

#     # หาทุก permutation ของ goal points เพื่อเปรียบเทียบการเดินทางทุกเส้นทาง
#     for seq in permutations(goal_points):
#         current_joint = start_joint  # เริ่มต้นที่ตำแหน่งเริ่มต้น

#         total_joint_change = 0  # เก็บการหมุน joint ทั้งหมด

#         # เปรียบเทียบเส้นทางและคำนวณการหมุน joint ในแต่ละ leg
#         for point in seq:
#             q1_sol, q2_sol, q3_sol = RRR.Inverse_Kinematics(point)

#             # คำนวณการเปลี่ยนแปลงของ joint (การหมุน)
#             joint_change = abs(current_joint.q1 - q1_sol[0]) + abs(current_joint.q2 - q2_sol[0]) + abs(current_joint.q3 - q3_sol[0])
#             total_joint_change += joint_change

#             # อัปเดต joint ปัจจุบัน
#             current_joint = Joint(q1=q1_sol[0], q2=q2_sol[0], q3=q3_sol[0])

#         # ตรวจสอบว่าเส้นทางนี้มีการหมุน joint น้อยที่สุดหรือไม่
#         if total_joint_change < min_joint_change:
#             min_joint_change = total_joint_change
#             best_sequence = seq

#     return best_sequence, min_joint_change

def TSP(RRR, start_joint, goal_points, tor_grid, obstacles):
    posible_paths = []

    num_point = len(goal_points)

    goal_joints = []
    for goal in goal_points:
        goal_joint = RRR.Inverse_Kinematics(goal)
        goal_joints.append(goal_joint)

        # check ว่า goal point อยู่นอกขอบเขตของหุ่นยนต์มั๊ย
        check_reach = is_reachable(goal_joint)
        if not check_reach:
            print("!!! Goal Point ", goal, " cannot reach !!!")
            return -1, -1
        
        # check ว่า goal point ชนสิ่งกีดขวางมั๊ย
        RRR.update_joint(goal_joint)
        goal_joint_pos = RRR.Forward_Kinematics()
        goal_joint_position = convert_goal_point_to_2d_list(goal_joint_pos)
        print("\nChecking for Collisions...")
        collision_detected = detect_collision(obstacles, goal_joint_position)
        if collision_detected:
            print("!!! Goal Point ", goal, " Collisions with obstacle !!!")
            return -1, -1

    for i in range(-1, num_point-1):
        for j in range(i+1, num_point):
            if i == -1:
                start = start_joint
                goal = goal_joints[j]
            else:
                start = goal_joints[i]
                goal = goal_joints[j]

            start_indices, goal_indices = map_joint_to_grid_indices(start, goal, M)

            # Run A* to find the path
            path = astar_torus(tor_grid, start_indices, goal_indices, M)
            
            if path:
                q1_dist, q2_dist, q3_dist = calculate_joint_distance(path, M)

                total_dist = q1_dist + q2_dist + q3_dist

                posible_paths.append(P2P_Path(i, j, start, path, total_dist))

            else:
                print("!!! Goal Point ", goal, " No path found !!!")
                return -1, -1
            
    if(num_point == 1):
        sequence = [[0, -1, 0]]

    elif(num_point == 2):
        # TSP (Brute Force)
        min_dist = 99999

        # start->goal1->goal2  
        dist = posible_paths[0].total_dist + posible_paths[2].total_dist    
        if dist < min_dist: 
            min_dist = dist
            sequence = [[0, -1, 0], [2, 0, 1]] # [posible_path_index, i, j]

        # start->goal2->goal1
        dist = posible_paths[1].total_dist + posible_paths[2].total_dist
        if dist < min_dist: 
            min_dist = dist
            sequence = [[1, -1, 1], [2, 1, 0]]

    elif(num_point == 3):
        # TSP (Brute Force)
        min_dist = 99999

        # start->goal1->goal2->goal3
        dist = posible_paths[0].total_dist + posible_paths[3].total_dist + posible_paths[5].total_dist       
        if dist < min_dist: 
            min_dist = dist
            sequence = [[0, -1, 0], [3, 0, 1], [5, 1, 2]] # [posible_path_index, i, j]

        # start->goal1->goal3->goal2
        dist = posible_paths[0].total_dist + posible_paths[4].total_dist + posible_paths[5].total_dist       
        if dist < min_dist: 
            min_dist = dist
            sequence = [[0, -1, 0], [4, 0, 2], [5, 2, 1]] # [posible_path_index, i, j]
        
        # start->goal2->goal1->goal3
        dist = posible_paths[1].total_dist + posible_paths[3].total_dist + posible_paths[4].total_dist       
        if dist < min_dist: 
            min_dist = dist
            sequence = [[1, -1, 1], [3, 1, 0], [4, 0, 2]] # [posible_path_index, i, j]

        # start->goal2->goal3->goal1
        dist = posible_paths[1].total_dist + posible_paths[5].total_dist + posible_paths[4].total_dist       
        if dist < min_dist: 
            min_dist = dist
            sequence = [[1, -1, 1], [5, 1, 2], [4, 2, 0]] # [posible_path_index, i, j]

        # start->goal3->goal1->goal2
        dist = posible_paths[2].total_dist + posible_paths[4].total_dist + posible_paths[3].total_dist       
        if dist < min_dist: 
            min_dist = dist
            sequence = [[2, -1, 2], [4, 2, 0], [3, 0, 1]] # [posible_path_index, i, j]

        # start->goal3->goal2->goal1
        dist = posible_paths[2].total_dist + posible_paths[5].total_dist + posible_paths[3].total_dist       
        if dist < min_dist: 
            min_dist = dist
            sequence = [[2, -1, 2], [5, 2, 1], [3, 1, 0]] # [posible_path_index, i, j]

    return posible_paths, sequence

#------------------------------------Plot & Animation---------------------------------#

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

    # Plot obstacles
    for obstacle in obstacles:
        u = np.linspace(0, 2 * pi, 100)
        v = np.linspace(0, pi, 100)
        x = obstacle[3] * np.outer(np.cos(u), np.sin(v)) + obstacle[0]
        y = obstacle[3] * np.outer(np.sin(u), np.sin(v)) + obstacle[1]
        z = obstacle[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle[2]
        ax.plot_surface(x, y, z, color='r', alpha=0.5)  # Semi-transparent obstacle

    # Add labels and legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
    # ax.text(joint_pos.PE.x, joint_pos.PE.y, joint_pos.PE.z, f"({joint_pos.PE.x:.2f}, {joint_pos.PE.y:.2f}, {joint_pos.PE.z:.2f})", color='orange')

    # Set axis limits to match workspace
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 6])

    # Show the plot
    plt.show()
    
def convert_path_to_radian(path):
    """
    Convert a list of tuples representing joint angles from range (0-99) to (0-pi) in radians.

    Parameters:
    - path: List of tuples [(q1, q2, q3), ...]

    Returns:
    - radian_path: List of Points [(x, y, z), ...] in radians
    """
    radian_path = []
    scale_factor = (2*pi) / M  # Conversion factor from 0-99 to 0-pi

    for joint_angles in path:
        q1, q2, q3 = joint_angles
        radian_path.append(Joint(
            q1 * scale_factor,  # แปลง q1 เป็น radians
            q2 * scale_factor,  # แปลง q2 เป็น radians
            q3 * scale_factor   # แปลง q3 เป็น radians
        ))

    return radian_path

def animate_path(path, obstacles, RRR, start_joint):
    # Prepare data
    goal_points = convert_path_to_radian(path)  # goal_points is a list of Point objects
    # print('goal_points:', goal_points)
    # goal_points = Joint(path_convert)
    # RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=start_joint.q1, q2=start_joint.q2, q3=start_joint.q3)
    RRR.update_joint(start_joint)

    # Create a figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize the plot
    def init():
        ax.clear()
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([0, 6])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        for obstacle in obstacles:
            u = np.linspace(0, 2 * pi, 100)
            v = np.linspace(0, pi, 100)
            x = obstacle[3] * np.outer(np.cos(u), np.sin(v)) + obstacle[0]
            y = obstacle[3] * np.outer(np.sin(u), np.sin(v)) + obstacle[1]
            z = obstacle[3] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle[2]
            ax.plot_surface(x, y, z, color='r', alpha=0.5)  # Semi-transparent obstacle

    # Update function for animation
    def update(frame):
        goal_point = goal_points[frame]
        RRR.update_joint(goal_point)  # Update robot position
        joint_pos = RRR.Forward_Kinematics()  # Get joint positions

        # Extract joint positions
        p1 = [float(joint_pos.P1.x), float(joint_pos.P1.y), float(joint_pos.P1.z)]
        p2 = [float(joint_pos.P2.x), float(joint_pos.P2.y), float(joint_pos.P2.z)]
        p3 = [float(joint_pos.P3.x), float(joint_pos.P3.y), float(joint_pos.P3.z)]
        pE = [float(joint_pos.PE.x), float(joint_pos.PE.y), float(joint_pos.PE.z)]

        # Extract coordinates for links
        x_coords = [p1[0], p2[0], p3[0], pE[0]]
        y_coords = [p1[1], p2[1], p3[1], pE[1]]
        z_coords = [p1[2], p2[2], p3[2], pE[2]]

        # Clear and replot
        ax.clear()
        init()
        ax.scatter(*p1, color='red', s=100, label='Base (P1)')
        ax.scatter(*p2, color='blue', s=100, label='Joint 1 (P2)')
        ax.scatter(*p3, color='green', s=100, label='Joint 2 (P3)')
        ax.scatter(*pE, color='purple', s=100, label='End Effector (PE)')
        ax.plot(x_coords, y_coords, z_coords, color='black', label='Robot Links')
        ax.legend()

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(goal_points), init_func=init, repeat=False)

    # Show the animation
    plt.show()

#------------------------------------------main----------------------------------------#
    
# # Chack RRR Class All Function
# def main():
#     RRR = RRR_Robot(l1=1, l2=1, l3=0.4, q1=0.1, q2=-0.5, q3=0.8)

#     # สร้างสิ่งกีดขวางแบบสุ่ม
#     obstacles = generate_random_obstacles(num_obstacles=3, max_radius=0.5, x_limit=(-3, 3), y_limit=(-3, 3), z_limit=(0, 6))
#     print("Generated Obstacles:", obstacles)

#     # check Forward_Kinematics 
#     goal_point = RRR.Forward_Kinematics()
#     print("goal point")
#     print("Px ", goal_point.PE.x)
#     print("Py ", goal_point.PE.y)
#     print("Pz ", goal_point.PE.z)

#     #check Inverse_Kinematic
#     goal_joint_space = RRR.Inverse_Kinematics(goal_point.PE)
#     print("goal joint space")
#     print("q1_sol ", goal_joint_space.q1)
#     print("q2_sol ", goal_joint_space.q2)
#     print("q3_sol ", goal_joint_space.q3)

#     RRR.update_joint(goal_joint_space)

#     position = RRR.Forward_Kinematics()
#     print("goal point")
#     print("Px ", position.PE.x)
#     print("Py ", position.PE.y)
#     print("Pz ", position.PE.z)

    
# # Check detect_collision Function
# def main():
#     # ตั้งค่าหุ่นยนต์
#     RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=-1, q2=1, q3=-0.5)

#     # แสดงข้อมูลสิ่งกีดขวาง
#     print("=" * 40)
#     print("Generated Obstacles:")
#     print("=" * 40)
#     for i, obs in enumerate(obstacles, start=1):
#         print(f"Obstacle {i}: Center=({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}), Radius={obs[3]:.2f}")
#     print("=" * 40)

#     # คำนวณ Forward Kinematics
#     print("\nCalculating Forward Kinematics...")
#     goal_point = RRR.Forward_Kinematics()
#     print(f"Goal Point (End Effector Position):")
#     print(f"Px: {float(goal_point.PE.x):.2f}, Py: {float(goal_point.PE.y):.2f}, Pz: {float(goal_point.PE.z):.2f}")
#     print("=" * 40)

#     # แปลงตำแหน่งข้อต่อเป็น 2D List
#     joint_positions = convert_goal_point_to_2d_list(goal_point)
#     print("\nJoint Positions (2D List):")
#     for i, pos in enumerate(joint_positions, start=1):
#         print(f"Joint {i}: x={float(pos[0]):.2f}, y={float(pos[1]):.2f}, z={float(pos[2]):.2f}")
#     print("=" * 40)

#     # ตรวจสอบการชน
#     print("\nChecking for Collisions...")
#     collision_detected = detect_collision(obstacles, joint_positions)
#     if collision_detected:
#         print("⚠️  Collision Detected!")
#     else:
#         print("✅  No Collision Detected.")
#     print("=" * 40)

#     # แสดงกราฟหุ่นยนต์และสิ่งกีดขวาง
#     print("\nDisplaying Robot and Obstacles...")
#     Show_plot(goal_point, obstacles)
#     print("Visualization Complete!")

# # Test the A* algorithm
# if __name__ == "__main__":
#     M = 100  # Grid size
#     grid = np.zeros((M, M, M), dtype=int)  # Empty grid
    
#     # Add obstacles (example)
#     grid[25][25][25] = 1
#     grid[50][50][50] = 1
#     grid[75][75][75] = 1
    
#     # Define start and goal positions in grid indices
#     start = (10, 10, 10)
#     goal = (90, 90, 90)
    
#     # Run A* algorithm
#     path = astar_torus(grid, start, goal, M)
#     print("Path found:", path)


# Test joint moving distance
# if __name__ == "__main__":
#     # Example path with joint angles (in grid indices)
#     path = [(99, 10, 0), (0, 11, 1), (1, 12, 2)]

#     # Calculate the distance for each joint
#     q1_dist, q2_dist, q3_dist = calculate_joint_distance(path, M)

#     print(f"Joint 1 distance: {q1_dist}")
#     print(f"Joint 2 distance: {q2_dist}")
#     print(f"Joint 3 distance: {q3_dist}")


# # Create toroidal grid
# def main():
#     RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=-1, q2=1, q3=-0.5)
#     tor_grid = get_occupancy_grid(RRR, obstacles)
    
#     # Save tor_grid to file
#     save_tor_grid(tor_grid, 'tor_grid.npy')


# Base Main - Check A* and Animation
# def main():

#     tor_grid = load_tor_grid('tor_grid.npy')    # Load Grid ที่มี Obstacle
#     #tor_grid = np.zeros((M, M, M), dtype=int)  # สร้าง Grid เปล่าแบบไม่มี Obstacle

#     # Case 1 สามารถหา path planning ได้
#     # start_joint = Joint(1, 1, -0.5)
#     # goal_point = Point(2, 1.5, 3)

#     # Case 2 สามารถหา path planning ได้
#     # start_joint = Joint(0.2, 3, -1)
#     # goal_point = Point(-1, 2, 4)

#     # Case 3 สามารถหา path planning ได้
#     start_joint = Joint(1, 1, -0.5)
#     goal_point = Point(-1, 2, 3)

#     # Case 4 ไม่สามารถ Reach the goal ได้
#     # start_joint = Joint(1, 1, -0.5)
#     # goal_point = Point(3, -3, 3)

#     # Create RRR Robot
#     RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=start_joint.q1, q2=start_joint.q2, q3=start_joint.q3)
#     goal_joint = RRR.Inverse_Kinematics(goal_point) # หา goal ใน joint space
#     print('goal_joint NING TIK',goal_joint)
#     print("goal_joint: ", goal_joint.q1, goal_joint.q2, goal_joint.q3)

#     check_reach = is_reachable(goal_joint)

#     if(check_reach):
#         # หาตำแหน่งแต่ละ Joint ที่ Goal point
#         RRR.update_joint(goal_joint)
#         goal_joint_pos = RRR.Forward_Kinematics()
#         goal_joint_position = convert_goal_point_to_2d_list(goal_joint_pos)

#         # ตรวจสอบการชนที่ Goal point
#         print("\nChecking for Collisions...")
#         collision_detected = detect_collision(obstacles, goal_joint_position)

#         # แสดงกราฟหุ่นยนต์และสิ่งกีดขวาง
#         print("\nDisplaying Robot and Obstacles...")
#         Show_plot(goal_joint_pos, obstacles)
#         print("Visualization Complete!")

#         # กรณีที่ Goal point ไม่ชนสิ่งกีดขวาง ให้ทำการหา Path Planning ไปยัง Goal point
#         if not collision_detected:

#             start_indices, goal_indices = map_joint_to_grid_indices(start_joint, goal_joint, M)

#             # Run A* to find the path
#             path = astar_torus(tor_grid, start_indices, goal_indices, M)
            
#             if path:
#                 print("Path found!")
#                 print(path)
                
#                 plot_path(path)
#                 q1_dist, q2_dist, q3_dist = calculate_joint_distance(path, M)

#                 print(f"Joint 1 distance: {q1_dist}")
#                 print(f"Joint 2 distance: {q2_dist}")
#                 print(f"Joint 3 distance: {q3_dist}")
                
#                 animate_path(path, obstacles, RRR , start_joint)

#                 check_goal = convert_path_to_radian([path[-1]])
#                 print("q1_goal_check", check_goal[0].q1)
#                 print("q2_goal_check", check_goal[0].q2)
#                 print("q3_goal_check", check_goal[0].q3)

#             else:
#                 print("No path found!")



def main():
    tor_grid = load_tor_grid('tor_grid.npy')    # Load Grid ที่มี Obstacle

    start_joint = Joint(1, 1, -0.5)
    
    goal_point = []
    goal_point.append(Point(2, 1.5, 3))
    goal_point.append(Point(-1, 2, 4))
    goal_point.append(Point(-1, 2, 3))

    RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=start_joint.q1, q2=start_joint.q2, q3=start_joint.q3)

    posible_paths, seqence = TSP(RRR, start_joint, goal_point, tor_grid, obstacles)

    if posible_paths != -1:
        for p in posible_paths:
            print("---------------------------------------------------------")
            print("i=", p.i, " j=", p.j, " start=", p.start.q1, ", ", p.start.q2, ", ", p.start.q3 , " path=", p.path, " total_dist=", p.total_dist)

        print("**** Optimal Moving Sequence ****")
        for s in seqence:
            if(s[1] == -1):
                print("Start ->", end=" ")
            else:
                print("Goal ", s[1]+1, " ->", end=" ")
            print("Goal ", s[2]+1)

            if s[1] > s[2]:
                posible_paths[s[0]].path.reverse()
            animate_path(posible_paths[s[0]].path, obstacles, RRR , posible_paths[s[0]].start)


if __name__ == '__main__':
    main()
    
    



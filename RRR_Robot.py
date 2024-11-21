import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from math import pi
import numpy as np

# Simulation parameters
M = 100
obstacles = [
    [1.5, -2, 3, 0.7],  # ทรงกลมที่จุด (1.75, 0.75, 0.6) รัศมี 0.3
    [2.5, 2, 4, 0.5],  # ทรงกลมที่จุด (1.75, 0.75, 0.6) รัศมี 0.3
    [-2.5, 0, 1.2, 0.5],   # ทรงกลมที่จุด (0.55, 1.5, 1.2) รัศมี 0.5
    [0, -1, 3, 0.5],       # ทรงกลมที่จุด (0, -1, 0.5) รัศมี 0.25
    [-1.75, 2.5 , 0.5, 0.5 , 0.8]       # ทรงกลมที่จุด (0, -1, 0.5) รัศมี 0.25
]

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


# def detect_collision(obstacles, joint_positions):
#     """
#     Check if the joint positions are close to any obstacles.
#     :param obstacles: List of obstacles in the form [[x, y, z, r], ...].
#     :param joint_positions: 2D list of joint positions [[x1, y1, z1], [x2, y2, z2], ...].
#     :return: 1 if there is a collision, 0 otherwise.
#     """
#     for joint_position in joint_positions:
#         joint_vec = np.array([float(coord) for coord in joint_position])  # Convert to float

#         for obstacle in obstacles:
#             obstacle_center = np.array([float(coord) for coord in obstacle[:3]])  # Convert to float
#             radius = float(obstacle[3])  # Convert to float

#             # Calculate distance
#             distance = np.linalg.norm(joint_vec - obstacle_center)

#             # Check if within the radius
#             if distance <= radius:
#                 print(f"Collision detected at joint position {joint_vec}\nwith obstacle {obstacle_center}.")
#                 return 1  # Collision detected

#     print("No collision detected.")
#     return 0  # No collision

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
        point = np.array([float(coord) for coord in point], dtype=float)

        line_vec = line_end - line_start
        point_vec = point - line_start

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

    

def main():
    # ตั้งค่าหุ่นยนต์
    RRR = RRR_Robot(l1=1.5, l2=1.5, l3=2, q1=-1, q2=1, q3=-0.5)

    # แสดงข้อมูลสิ่งกีดขวาง
    print("=" * 40)
    print("Generated Obstacles:")
    print("=" * 40)
    for i, obs in enumerate(obstacles, start=1):
        print(f"Obstacle {i}: Center=({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}), Radius={obs[3]:.2f}")
    print("=" * 40)

    # คำนวณ Forward Kinematics
    print("\nCalculating Forward Kinematics...")
    goal_point = RRR.Forward_Kinematics()
    print(f"Goal Point (End Effector Position):")
    print(f"Px: {float(goal_point.PE.x):.2f}, Py: {float(goal_point.PE.y):.2f}, Pz: {float(goal_point.PE.z):.2f}")
    print("=" * 40)

    # แปลงตำแหน่งข้อต่อเป็น 2D List
    joint_positions = convert_goal_point_to_2d_list(goal_point)
    print("\nJoint Positions (2D List):")
    for i, pos in enumerate(joint_positions, start=1):
        print(f"Joint {i}: x={float(pos[0]):.2f}, y={float(pos[1]):.2f}, z={float(pos[2]):.2f}")
    print("=" * 40)

    # ตรวจสอบการชน
    print("\nChecking for Collisions...")
    collision_detected = detect_collision(obstacles, joint_positions)
    if collision_detected:
        print("⚠️  Collision Detected!")
    else:
        print("✅  No Collision Detected.")
    print("=" * 40)

    # แสดงกราฟหุ่นยนต์และสิ่งกีดขวาง
    print("\nDisplaying Robot and Obstacles...")
    Show_plot(goal_point, obstacles)
    print("Visualization Complete!")



if __name__ == '__main__':
    main()

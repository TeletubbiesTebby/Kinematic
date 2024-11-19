import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the points in 3D space
point1 = [1, 2, 0]  # (x1, y1, z1)
point2 = [4, 5, 6]  # (x2, y2, z2)

# Extract coordinates for plotting
x_coords = [point1[0], point2[0]]
y_coords = [point1[1], point2[1]]
z_coords = [point1[2], point2[2]]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points with a larger radius (size parameter `s`)
ax.scatter(*point1, color='red', s=100, label='Point 1')  # Radius approx. 0.1
ax.scatter(*point2, color='blue', s=100, label='Point 2')  # Radius approx. 0.1

# Plot the line connecting the points
ax.plot(x_coords, y_coords, z_coords, color='green', label='Connecting Line')

# Add labels and legend
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()

# Fix the origin at (0, 0, 0) and z starting from 0
max_range = max(
    max(abs(point1[0]), abs(point2[0])),
    max(abs(point1[1]), abs(point2[1])),
    max(abs(point1[2]), abs(point2[2]))
)
ax.set_xlim([-max_range, max_range])  # X-axis centered
ax.set_ylim([-max_range, max_range])  # Y-axis centered
ax.set_zlim([0, max_range])          # Z-axis starts at 0 and goes positive

# Show the plot and keep it open until manually closed
plt.show(block=True)

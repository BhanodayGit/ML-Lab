import matplotlib.pyplot as plt
import numpy as np

# Define the data points for each class
positive_points = np.array([[4, 1],
                            [4, -1],
                            [6, 0]])
negative_points = np.array([[1, 0],
                            [0, 1],
                            [0, -1]])

# Extract the x and y coordinates for each class
pos_x, pos_y = positive_points[:, 0], positive_points[:, 1]
neg_x, neg_y = negative_points[:, 0], negative_points[:, 1]

# Create a new figure for the plot
plt.figure(figsize=(8, 6))

# Plot the points
plt.scatter(pos_x, pos_y, color="blue", marker="o", s=100, label="Positive (+1)")
plt.scatter(neg_x, neg_y, color="red", marker="x", s=100, label="Negative (-1)")

# Draw the optimal hyperplane (vertical line at x = 2.5)
plt.axvline(x=2.5, color="green", linestyle="solid", linewidth=2, label="Optimal Hyperplane (x = 2.5)")

# Draw the margin boundaries (vertical lines at x = 1 and x = 4)
plt.axvline(x=1.0, color="black", linestyle="dashed", linewidth=1.5, label="Margin Boundary (x = 1)")
plt.axvline(x=4.0, color="black", linestyle="dashed", linewidth=1.5, label="Margin Boundary (x = 4)")

# Set the plot labels and title
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("SVM Optimal Hyperplane and Margin Boundaries")

# Set the axis limits for clarity
plt.xlim(-1, 8)
plt.ylim(-3, 3)

# Turn on the grid
plt.grid(True)

# Display legend
plt.legend()

# Show the plot
plt.show()

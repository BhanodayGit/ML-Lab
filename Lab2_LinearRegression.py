import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def estimate_coef(x, y):
    n = np.size(x)

    # Mean values
    m_x = np.mean(x)
    m_y = np.mean(y)
    m_xy = np.mean(x * y)
    m_x2 = np.mean(x ** 2)

    # Calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # Calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    # Print mean values
    print(f"Mean X: {m_x:.2f}")
    print(f"Mean Y: {m_y:.2f}")
    print(f"Mean XY: {m_xy:.2f}")
    print(f"Mean X^2: {m_x2:.2f}")

    return b_0, b_1

def calculate_error(x, y, b):
    y_pred = b[0] + b[1] * x
    errors = y - y_pred  # Residual errors
    mse = np.mean(errors ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error

    print(f"\nMean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="m", marker="o", s=30, label="Data Points")

    # Predicted response vector
    y_pred = b[0] + b[1] * x

    # Plot regression line
    plt.plot(x, y_pred, color="g", label="Regression Line")

    # Labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

def main():
    df = pd.read_csv("linear_regression_data.csv")

    # Convert data to numpy arrays
    x = df['X'].values
    y = df['Y'].values

    # Estimate coefficients
    b = estimate_coef(x, y)

    print(f"\nEstimated coefficients:\nb_0 = {b[0]:.2f} \nb_1 = {b[1]:.2f}")

    # Calculate and print errors
    calculate_error(x, y, b)

    # Plot regression line
    plot_regression_line(x, y, b)

main()

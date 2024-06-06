import numpy as np
import matplotlib.pyplot as plt

class Interpolation:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.n = len(x_data)

    def lagrange_interpolation(self, x):
        y_interp = 0
        for i in range(self.n):
            term = self.y_data[i]
            for j in range(self.n):
                if j != i:
                    term *= (x - self.x_data[j]) / (self.x_data[i] - self.x_data[j])
            y_interp += term
        return y_interp

    def divided_diff_coefficients(self):
        # Compute divided differences table
        coefficients = np.zeros((self.n, self.n))
        coefficients[:,0] = self.y_data

        for j in range(1, self.n):
            for i in range(self.n - j):
                coefficients[i, j] = (coefficients[i+1, j-1] - coefficients[i, j-1]) / (self.x_data[i+j] - self.x_data[i])

        return coefficients[0]

    def newton_interpolation(self, x):
        coefficients = self.divided_diff_coefficients()
        y_interp = coefficients[0]
        temp = 1

        for i in range(1, self.n):
            temp *= (x - self.x_data[i-1])
            y_interp += coefficients[i] * temp

        return y_interp

# Data given
x_data = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_data = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Creating interpolation object
interpolator = Interpolation(x_data, y_data)

# Testing the interpolations
x_values = np.linspace(5, 40, 100)  # Values for plotting

# Lagrange Interpolation
lagrange_y_values = [interpolator.lagrange_interpolation(x) for x in x_values]

# Newton Interpolation
newton_y_values = [interpolator.newton_interpolation(x) for x in x_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, lagrange_y_values, label='Lagrange Interpolation')
plt.plot(x_values, newton_y_values, label='Newton Interpolation')
plt.scatter(x_data, y_data, color='red', label='Data Points')
plt.title('Interpolation of Time to Failure vs Tension')
plt.xlabel('Tension (kg/mm^2)')
plt.ylabel('Time to Failure (hours)')
plt.legend()
plt.grid(True)
plt.show()

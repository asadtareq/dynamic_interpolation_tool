import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Lagrange Interpolation
# -------------------------
def lagrange(x, y, xp):
    n = len(x)
    yp = 0
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                p *= (xp - x[j]) / (x[i] - x[j])
        yp += p * y[i]
    return yp

# -------------------------
# Input Data
# -------------------------
n = int(input("Enter number of data points: "))

x = []
y = []

for i in range(n):
    xi = float(input(f"x[{i}]: "))
    yi = float(input(f"y[{i}]: "))
    x.append(xi)
    y.append(yi)

x = np.array(x)
y = np.array(y)

# -------------------------
# Prediction point
# -------------------------
xp = float(input("\nEnter value to predict: "))

# -------------------------
# Lagrange Calculation
# -------------------------
yp_lagrange = lagrange(x, y, xp)

# -------------------------
# Curve Fitting (Polynomial)
# -------------------------
degree = int(input("Enter degree for curve fitting: "))
coeff = np.polyfit(x, y, degree)
poly = np.poly1d(coeff)
yp_fit = poly(xp)

# -------------------------
# Error Calculation
# -------------------------

# Lagrange error (should be near zero if perfect interpolation)
y_lagrange_all = [lagrange(x, y, xi) for xi in x]
error_lagrange = np.mean(abs(y - y_lagrange_all))

# Curve fitting error
y_fit_all = poly(x)
error_fit = np.mean(abs(y - y_fit_all))

# -------------------------
# Output Results
# -------------------------
print("\n========== RESULTS ==========")
print(f"Lagrange Prediction: {yp_lagrange:.6f}")
print(f"Curve Fit Prediction: {yp_fit:.6f}")

# -------------------------
# Comparison Table
# -------------------------
print("\n========== COMPARISON ==========")
print("Method\t\tPrediction\tError")

print(f"Lagrange\t{yp_lagrange:.6f}\t{error_lagrange:.6f}")
print(f"Curve Fit\t{yp_fit:.6f}\t{error_fit:.6f}")

# -------------------------
# Plot
# -------------------------
x_plot = np.linspace(min(x), max(x), 100)

y_lagrange_plot = [lagrange(x, y, xi) for xi in x_plot]
y_fit_plot = poly(x_plot)

plt.figure(figsize=(8,5))

# Original data
plt.scatter(x, y, label="Data Points")

# Methods
plt.plot(x_plot, y_lagrange_plot, label="Lagrange", linewidth=2)
plt.plot(x_plot, y_fit_plot, linestyle="--", label="Curve Fit", linewidth=2)

# Predictions
plt.scatter(xp, yp_lagrange, label="Predicted (Lagrange)")
plt.scatter(xp, yp_fit, label="Predicted (Fit)")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolation vs Curve Fitting")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

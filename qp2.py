import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from time import time
# Definirea constantelor functiei obiectiv: f = ax1^2 + bx2^2
a = 2
b = 3

# Definirea gradientului functiei Lagrangeana
def lagrangian_gradient(x, y, lambda1, lambda2):
    # Gradientul functiei Lagrangeana este dat de gradientul functiei obiectiv si a produselor scalar intre coeficientii de Lagrange si constrangeri
    grad_x = 2 * a * x - lambda1 - lambda2
    grad_y = 2 * b * y - lambda1
    grad_lambda1 = 100 - x - y
    grad_lambda2 = 40 - x
    return np.array([grad_x, grad_y, grad_lambda1, grad_lambda2])

def hessian():
    # Hessianul functiei Lagrangeana este dat de matricea Hessiana a functiei obiectiv si de matricea Hessiana a produselor scalar intre coeficientii de Lagrange si constrangeri
    return np.array([
        [2 * a, 0, -1, -1],
        [0, 2 * b, -1, 0],
        [-1, -1, 0, 0],
        [-1, 0, 0, 0]
    ])

# Implementarea metodei Newton pentru rezolvarea problemei de optimizare
def newton_method(initial_guess, tol=1e-6, max_iter=100):
    x, y, lambda1, lambda2 = initial_guess
    history = [[x, y]]
    for _ in range(max_iter):
        grad = lagrangian_gradient(x, y, lambda1, lambda2)
        hess = hessian()
        try:
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            print("Singular Hessian encountered.")
            break
        x += delta[0]
        y += delta[1]
        lambda1 += delta[2]
        lambda2 += delta[3]
        history.append([x, y])
        if np.linalg.norm(delta) < tol:
            break
    return np.array([x, y, lambda1, lambda2]), history

# Definirea valorilor initiale si rezolvarea problemei de optimizare
initial_guess = np.array([50, 50, 0, 0])

t1 = time()
result_newton, history = newton_method(initial_guess)
x_newton, y_newton, _, _ = result_newton
t2 = time()
duration = t2 - t1
print("Time taken for Newton's method:", duration)
# Grafic evolutie solutie
history = np.array(history)
fig, ax = plt.subplots()
ax.plot(history[:, 0], history[:, 1], 'o-')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Evolutia solutiei in spatiu')
plt.show()

# Definirea constrangerilor problemei de optimizare
def objective(X):
    x, y = X
    return a * x**2 + b * y**2

constraints = [
    {'type': 'ineq', 'fun': lambda X: X[0] + X[1] - 100},
    {'type': 'ineq', 'fun': lambda X: X[0] - 40}
]

bounds = [(0, None), (0, None)]
t1 = time()
result_scipy = minimize(objective, [50, 50], method='SLSQP', bounds=bounds, constraints=constraints)
t2 = time()
duration = t2 - t1
print("Time taken for SciPy minimize:", duration)
x_scipy, y_scipy = result_scipy.x

print("Newton's method result:", x_newton, y_newton)
print("SciPy minimize result:", x_scipy, y_scipy)

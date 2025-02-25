# Optimization-Lagrange-SciPy

## Overview
This project focuses on optimizing resource allocation in a data center using Lagrange multipliers and SciPy’s `minimize` function. The goal is to minimize energy consumption for two large-scale projects while satisfying hardware constraints.

## Problem Formulation
- **Objective Function:** Minimize the quadratic energy consumption function:  
  \[ Z = 2x_1^2 + 3x_2^2 \]  
  where:
  - \( x_1 \) represents nodes allocated for Project A.
  - \( x_2 \) represents nodes allocated for Project B.

- **Constraints:**  
  - \( x_1 + x_2 \leq 100 \) (Total nodes available constraint)
  - \( x_1 \geq 40 \) (Minimum requirement for Project A)

## Methods Used
### **Newton’s Method with Lagrange Multipliers**
- Implements the Lagrangian function:  
  \[ L(x, \lambda) = f(x) + \lambda_1 (100 - x_1 - x_2) + \lambda_2 (40 - x_1) \]
- Computes the **gradient** and **Hessian matrix** to iteratively solve for the optimal values.

### **SciPy Optimization (`SLSQP`)**
- Uses the `minimize` function from SciPy with Sequential Least Squares Programming (SLSQP) to solve the constrained optimization problem efficiently.

## Results
- **Newton’s Method:** Achieved optimal resource allocation with minimal computational time.
- **SciPy Optimization:** Validated results obtained via Newton’s method.
- **Performance:** Newton’s method ran in ~0.0001s, while SciPy took ~0.0018s.

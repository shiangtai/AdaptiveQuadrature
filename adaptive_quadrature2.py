import numpy as np
import matplotlib.pyplot as plt

# List to store the points where the function is evaluated
evaluation_points = []

def func(x):
    evaluation_points.append(x)
    return 4 * np.sqrt(1 - x * x)

def integrate(a, b, tol=1e-10, maxint=500):
    eps = np.finfo(float).eps
    if tol < eps:
        tol = eps

    fa = func(a)
    fb = func(b)
    m = 0.5 * (a + b)
    fm = func(m)
    return adaptive_simpson(a, b, fa, fm, fb, tol, 0, maxint)

def adaptive_simpson(a, b, fa, fm, fb, tol, cnt, maxint):
    m = 0.5 * (a + b)
    h = 0.25 * (b - a)
    fml = func(a + h)
    fmr = func(b - h)
    
    # Simpson's method between a and b
    i1 = h / 1.5 * (fa + 4.0 * fm + fb)
    # Divide the interval into 2 halves and do Simpson's integral again
    i2 = h / 3.0 * (fa + 4.0 * (fml + fmr) + 2.0 * fm + fb)
    
    if np.abs(i1 - i2) <= np.abs(tol):
        return i2
    else:
        if cnt < maxint:
            cnt += 1
            return adaptive_simpson(a, m, fa, fml, fm, tol, cnt, maxint) + adaptive_simpson(m, b, fm, fmr, fb, tol, cnt, maxint)
        else:
            print("Maximum iteration reached:", cnt)
            return 0

def main():
    precisions = [1e-2]
    for pre in precisions:
        global evaluation_points
        evaluation_points = []
        result = integrate(0, 1, pre)
        print(f"pi = {result:.18f} with precision = {pre:.3e} intervals = {len(evaluation_points)}")
        
        # Plotting the evaluated points
        plt.figure()
        x_vals = np.array(evaluation_points)
        y_vals = func(x_vals) /4
        plt.scatter(x_vals, y_vals, edgecolors='red', facecolors='none', marker='o', label='Evaluated Points')
        
        # Plotting the quarter circle for comparison
        theta = np.linspace(0, np.pi/2, 1000)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        plt.plot(x_circle, y_circle, label='Quarter Circle')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Function Evaluations for tol = {pre:.3e}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    main()

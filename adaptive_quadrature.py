import numpy as np

count = 0  # This is to record how many times func() is called

def func(x):
    global count
    count += 1
    return 4 * np.sqrt(1 - x * x)

def adaptive_simpson(a, b, fa, fm, fb, tol, cnt, maxint):
    m = (a + b) * 0.5
    h = (b - a) * 0.25
    fml = func(a + h)
    fmr = func(b - h)
    #simpson's method between a and b
    i1 = h / 1.5 * (fa + 4.0 * fm + fb)  # (b-a)/6*(fa+4.0*fm+fb)
    #divide the interval into 2 halves and do simpson's integral again
    i2 = h / 3.0 * (fa + 4.0 * (fml + fmr) + 2.0 * fm + fb)  # (b-a)/12*(fa+4.0*(fml+fmr)+2.0*fm+fb)

    if abs(i1 - i2) <= abs(tol):
        return i2  # If further division does not improve accuracy, return the value
    else:
        if cnt < maxint:
            cnt += 1
            return (adaptive_simpson(a, m, fa, fml, fm, tol, cnt, maxint) + 
                    adaptive_simpson(m, b, fm, fmr, fb, tol, cnt, maxint))
        else:
            print(f"Maximum iteration reached: {cnt}")
    return 0

def integrate(a, b, tol=1e-10, maxint=500):
    eps = np.finfo(float).eps #machine precision limit
    if tol < eps:
        tol = eps

    m = 0.5 * (a + b)
    fa = func(a)
    fb = func(b)
    fm = func(m)
    return adaptive_simpson(a, b, fa, fm, fb, tol, 0, maxint)

if __name__ == "__main__":
    precisions = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    for pre in precisions:
        count = 0
        tot = integrate(0, 1, pre)
        print(f"pi={tot:.18f} with precision={pre:.3e} intervals={count}")

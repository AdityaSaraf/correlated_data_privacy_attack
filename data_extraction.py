from math import exp
import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import LineString, Point

def A(q, r, eps):
    er = exp(eps)*r
    root = math.sqrt((q - er)**2 + 4*exp(eps)*(1-q)*(1-r))
    return abs(q - er) * root

def B(q, r, eps):
    return (q-exp(eps)*r)**2 + 2 * exp(eps) * (1-q)*(1-r)

def C(q, r, eps):
    return 2*(1-q)**2 * exp(eps)

# Represents A', the result of switching q and r in A
def A_alt(q, r, eps):
    return A(r, q, eps)

# Represents B', the result of switching q and r in B
def B_alt(q, r, eps):
    return B(r, q, eps)

# Represents C', the result of switching q and r in B
def C_alt(q, r, eps):
    return C(r, q, eps)

# Currently only returns the middle calculation, where neither rho_0 nor rho_1 are 0.5
# Returns (rho_0, rho_1)
def min_exp_noise(q, r, eps):
    e = exp(eps)
    a = A(q, r, eps)
    b = B(q, r, eps)
    c = C(q, r, eps)
    a_ = A_alt(q, r, eps)
    b_ = B_alt(q, r, eps)
    c_ = C_alt(q, r, eps)
    if (e >= q/r):
        line1 = LineString([ (1-(b+a)/(2*c), 0.5), (0.5, c/(2*(b+a))) ])
    else: # e < q/r
        line1 = LineString([ (1-(b-a)/(2*c), 0.5), (0.5, c/(2*(b-a))) ])
    if (e >= r/q):
        line2 = LineString([ (c_/(2*(b_+a_)),0.5), (0.5,1-(b_+a_)/(2*c_)) ])
    else: # e < r/q
        line2 = LineString([ (c_/(2*(b_-a_)),0.5), (0.5,1-(b_-a_)/(2*c_)) ])
    
    (x_1, y_1), (x_2, y_2) = line1.coords
    slope1 = (y_2-y_1)/(x_2 - x_1)
    (x_1, y_1), (x_2, y_2) = line2.coords
    slope2 = (y_2-y_1)/(x_2 - x_1)
    line_top = LineString([(0, 0.5), (0.5, 0.5)])
    line_right = LineString([(0.5, 0), (0.5, 0.5)])
    if abs(slope2) > abs(slope1):
        intsect_top = line2.intersection(line_top)
        intsect_right = line1.intersection(line_right)
    else:
        intsect_top = line1.intersection(line_top)
        intsect_right = line2.intersection(line_right)
    intsect = line1.intersection(line2)
    pts = [intsect, intsect_top, intsect_right]
    exp_noise = np.fromiter(map(lambda pt: pt.x * r/(q+r) + pt.y * q/(q+r), pts), dtype=float)
    print(exp_noise)
    min_pt = pts[np.argmin(exp_noise)]
    return min_pt.x, min_pt.y
    # rho_0, rho_1 = intsect.x, intsect.y
    # return (rho_0, rho_1)

if __name__ == "__main__":
    lines = []
    with open('data/yoga/Y1.med', 'r') as f:
        lines = f.read().splitlines()
        lines = [item.split(' ')[1] for item in lines]
    data = np.array(lines, dtype=float)
    # data = data[0:len(data):2]
    avg = np.average(data)
    binary_dat = [item > avg for item in data] # state 0 (aka False) indicates rest state
    transition_counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}

    for i in range(len(binary_dat) - 1):
        from_state = binary_dat[i]
        to_state = binary_dat[i+1]
        transition_counts[(from_state, to_state)] += 1

    q = transition_counts[(False, True)]/(transition_counts[(False, True)] + transition_counts[(False, False)])
    r = transition_counts[(True, False)]/(transition_counts[(True, False)] + transition_counts[(True, True)])
    print('(q, r):', (q,r))
    print('Original sum: ', np.sum(binary_dat))
    q, r = 0.02, 0.02
    for eps in [0.2, 0.5, 1, 2]:
        rho_0, rho_1 = min_exp_noise(q, r, eps)
        print('eps:', eps, '(rho_0, rho_1):', (rho_0, rho_1))
        data = np.array(binary_dat, dtype=bool)
        for i, val in enumerate(data):
            if val: # 1 state
                if np.random.rand() < rho_1:
                    data[i] = not val
            else: # 0 state
                if np.random.rand() < rho_0:
                    data[i] = not val
        print('sum: ', np.sum(data))

    print(transition_counts)
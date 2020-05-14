from math import exp
import numpy as np
import matplotlib.pyplot as plt
import hmms
import sys
import math

def A(q, r, eps):
    er = exp(eps)*r
    root = math.sqrt((q**2 - er**2)**2 + 4*(1-q-r) * exp(eps)*(q-er)**2)
    return 0.5*root - exp(eps)*(1-q-r)-0.5*(q**2 + er**2)

def B(q, r, eps):
    return exp(eps)*(1-q)**2

# Represents A', the result of switching q and r in A
def A_alt(q, r, eps):
    return A(r, q, eps)

# Represents B', the result of switching q and r in B
def B_alt(q, r, eps):
    return B(r, q, eps)

# Currently only returns the middle calculation, where neither rho_0 nor rho_1 are 0.5
# Returns (rho_0, rho_1)
def min_exp_noise(q, r, eps):
    a = A(q, r, eps)
    b = B(q, r, eps)
    a_ = A_alt(q, r, eps)
    b_ = B_alt(q, r, eps)

    rho_0 = ((a + b)*b_)/(b*b_ - a*a_)
    rho_1 = 1 + ((a + b)*a_)/(b*b_ - a*a_)
    return (rho_0, rho_1)

lines = []
with open('data/yoga/Y1.med', 'r') as f:
    lines = f.read().splitlines()
    lines = [item.split(' ')[1] for item in lines]
    print('test')
data = np.array(lines, dtype=float)
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
q, r = 0.2, 0.3
for eps in [0.2, 0.5, 1, 2]:
    rho_0, rho_1 = min_exp_noise(q, r, eps)
    print('eps:', eps, '(rho_0, rho_1):', (rho_0, rho_1))


print(transition_counts)
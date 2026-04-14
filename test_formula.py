import math
eps=1e-8
w_n=1000
d_ema=0.038
mu=0.001
sigma=0.0116
z = (abs(d_ema) - mu) / sigma
f = max(0, z - 1.0)
print(f"z={z}, f={f}")

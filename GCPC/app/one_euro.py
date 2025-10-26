# -*- coding: utf-8 -*-
# Minimal OneEuro filter for smoothing 2D landmarks
import math, time

class LowPass:
    def __init__(self, alpha, x0=None):
        self.y = x0
        self.a = alpha
    def apply(self, x):
        if self.y is None:
            self.y = x
        else:
            self.y = self.a * x + (1.0 - self.a) * self.y
        return self.y

def alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau/dt)

class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.t_prev = None
        self.dx_filt = None
        self.x_filt = None
    def apply(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            dt = 1/60.0
        else:
            dt = max(1e-4, t - self.t_prev)
        self.t_prev = t
        # derivative
        dx = x if self.x_prev is None else (x - self.x_prev) / dt
        self.x_prev = x
        # smooth derivative
        if self.dx_filt is None:
            self.dx_filt = LowPass(alpha(self.d_cutoff, dt), dx)
            dx_hat = dx
        else:
            self.dx_filt.a = alpha(self.d_cutoff, dt)
            dx_hat = self.dx_filt.apply(dx)
        # dynamic cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        if self.x_filt is None:
            self.x_filt = LowPass(alpha(cutoff, dt), x)
            return x
        self.x_filt.a = alpha(cutoff, dt)
        return self.x_filt.apply(x)

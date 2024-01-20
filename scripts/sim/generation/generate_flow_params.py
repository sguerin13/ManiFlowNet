import os
import numpy as np


def log_linear(low, high, n_increments):
    return np.logspace(low, high, n_increments, endpoint=True)


if __name__ == "__main__":
    # Reynolds Numbers: range 100 - 10^6
    Re_No = log_linear(2, 5, 2000)

    # Roughness Factor Epsilon: 10^-6 - .05
    Eps = log_linear(-6, np.log10(0.05), 2000)

    # Viscosity
    Visc = log_linear(np.log10(0.0003), np.log10(1.4), 2000)

    # Density
    Density = np.linspace(600, 3120, 2000, endpoint=True)

    # Pressure 1 atm - 200 psi
    P = np.linspace(1e5, 1.38e6, 2000, endpoint=True)

    # plt.scatter(range(1,51),Re_No)

    params = np.vstack((Re_No, Eps, Visc, Density, P))
    params = params.T

    np.savetxt(os.path.join("outputs", "flow_params.csv"), params, delimiter=",")

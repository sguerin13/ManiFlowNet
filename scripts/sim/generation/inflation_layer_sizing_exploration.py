from sim.generation.y_plus_calcs import *
from generate_flow_params import log_linear
import numpy as np
import matplotlib.pyplot as plt


'''
script to test the inflation layer sizing logic
'''
if __name__ == "__main__":
    # Reynolds Numbers: range 100 - 10^6
    Re_No = list(log_linear(3,5,50))
    # Roughness Factor Epsilon: 10^-6 - .05
    Eps = list(log_linear(-6,np.log10(.05),50))

    D = list(np.linspace(.01,.2,num = 20,endpoint=True))

    rho_water = 1000 # kg/m^3
    mew = .000889
    mew_kin = mew/rho_water

    V_mat = np.zeros((len(Re_No),len(Eps),len(D)))
    bl_height_mat = np.zeros((len(Re_No),len(Eps),len(D)))


    for i in range(len(Re_No)):
        for j in range(len(Eps)):
            for k in range(len(D)):
                f_d = colbrook_eq(Eps[j],Re_No[i])
                
                V = Re_No[i]*mew/(rho_water*D[k])
                V_mat[i,j,k] = V
                bl_height = inf_layer_height(f_d,V,rho_water,mew_kin,300)
                bl_height_mat[i,j,k] = bl_height


    # ok now lets probe the extremes of the space
    plt.figure(figsize  = (10,10))
    for i in range(len(Re_No)):

        plt.plot(Re_No,bl_height_mat[:,i,-1]*1000)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Re #')
        plt.ylabel('Boundary layer height (mm)')
        plt.title('Boundary Layer, Y+ = 300, various Eps Values, Pipe Diameter: 200 mm')

    # plt.plot([1000,100000],[1,1])
    # plt.plot([1000,100000],[.1,.1])

    plt.plot([1000,100000],[20,20])
    plt.plot([1000,100000],[2,2])

    plt.show()
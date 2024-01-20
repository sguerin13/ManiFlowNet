import math

def colbrook_eq(Eps,Re):
    # using swamee-jain equation
    log_term = Eps/3.7 + 5.4/(Re**.9)
    f_d = .25/((math.log10(log_term))**2)
    return f_d

def laminar_f_d(Re):
    f_d = float(64)/Re
    return f_d

def calc_f_d(Re,Eps):
    
    if (Re>3000):
        f_d = colbrook_eq(Eps,Re)
    
    else:
        f_d = laminar_f_d(Re)
    
    return f_d

def calc_Re(D,V,rho,mew):
    Re = rho*V*D/mew
    return Re

def calc_V(Re,rho,mew,D_in,D_out,A_in,A_out):
    '''

    Impose some velocity limits, since incompressible flow, need to ensure
    velocity at any point doesn't exceed 100 m/s, this can be done by comparing
    inlet and outlet areas

    '''
    n_out = float(len(D_out))
    n_in = float(len(D_in))

    D = sum(D_in)/n_in # average inlet diameter
    V = Re*mew/(rho*D)
    Re = Re
    # impost some velocity bounds
    if V > 100.0:
        V = 100.0
        Re = calc_Re(D,V,rho,mew)

    if V < .01:
        V = .01
        Re = calc_Re(D,V,rho,mew)

    # make sure the outlet velocity doesn't exceed Mach .3
    Q_in = A_in*V
    D_out_min = min(D_out)
    A_out_min = D_to_A(D_out_min)
    V_out_max = V_out_max_from_V(A_in,V,A_out_min,n_out)

    if V_out_max > 100.0:
        V_out_max = 100.0
        V = V_from_V_out_max(V_out_max,A_out_min,A_in,n_out)
        Re = calc_Re(D,V,rho,mew)
    
    return V, V_out_max, Re

def V_out_max_from_V(A_in,V,A_out_min,n_out):
    Q_in = A_in*V
    return (Q_in/n_out)/A_out_min

def V_from_V_out_max(V_out_max,A_out_min,A_in,n_out):
    return V_out_max*A_out_min*n_out/A_in

def D_to_A(D):
    return 0.25*math.pi*(D**2)

def inf_layer_height(f_d,V,rho,mew_kin,y_plus_tar):
    '''

    automatic inflation layer sizing based on the flow parameters

    '''
    # fanning friction factor is 1/4 of darcy friction factor
    f_fanning = f_d/float(4)
    tau = .5*rho*f_fanning*(V**2)
    y = y_plus_tar*mew_kin/(math.sqrt(tau/rho))
    return y

def size_boundary_layer(D,Re,Eps,V,V_out_max,rho,mew,mew_kin,D_in,D_out,A_in,A_out):
    '''
    Need to size the boundary layer based on worst case scenario, this will
    be based on the area with the maximum local velocity. If velocity at an
    outlet is greater than the inlet velocity then the Re # and velocity
    from the outlet should be used.


    '''
    y_tar = float(100)
    if V > V_out_max:
        V_max = 'inlet'
        f_d = calc_f_d(Re,Eps)
    else:
        V_max = 'outlet'
        D_min = min(D_out)
        Re_out = calc_Re(D_min,V_out_max,rho,mew)
        f_d = calc_f_d(Re_out,Eps)

    if V_max == 'inlet':
        inf_y = inf_layer_height(f_d,V,rho,mew_kin,y_tar)
    else:
        inf_y = inf_layer_height(f_d,V_out_max,rho,mew_kin,y_tar)
    
    if (inf_y < .01*D):
        y_tar =  float(200)
        if V_max == 'inlet':
            inf_y = inf_layer_height(f_d,V,rho,mew_kin,y_tar)
        else:
            inf_y = inf_layer_height(f_d,V_out_max,rho,mew_kin,y_tar)
        
        while (inf_y < .01*D): # mesh is probably too fine
            # if we are still outside the viable range
            # tune down the velocity, dropping the reynolds number
            # until a y+ of 300 is in the viable range
            if V_max == 'inlet':
                V = V*.9
                Re = calc_Re(D,V,rho,mew) 
                f_d = calc_f_d(Re,Eps)
                inf_y = inf_layer_height(f_d,V,rho,mew_kin,y_tar)
            else:
                V_out_max = V_out_max*.9
                Re_out = calc_Re(D_min,V_out_max,rho,mew)
                f_d = calc_f_d(Re_out,Eps)
                inf_y = inf_layer_height(f_d,V_out_max,rho,mew_kin,y_tar)
                # update the inlet velocity and reynolds #
                A_out_min = D_to_A(D_min)
                n_out = float(len(D_out))
                V = V_from_V_out_max(V_out_max,A_out_min,A_in,n_out)
                Re = calc_Re(D,V,rho,mew)
            
    # if we are in this region of Y+, we don't need to worry about exceeding
    # Mach .3
    if (inf_y > .1*D): # boundary layer too large, lets lower the target Y+
        y_tar =  float(30)
        if V_max == 'inlet':
            inf_y = inf_layer_height(f_d,V,rho,mew_kin,y_tar)
        else:
            inf_y = inf_layer_height(f_d,V_out_max,rho,mew_kin,y_tar)
    
    if (inf_y > .1*D): # boundary layer still too large
        # need to use a different wall function Y+ < 5
        y_tar = float(3)
        if V_max == 'inlet':
            inf_y = inf_layer_height(f_d,V,rho,mew_kin,y_tar)
        else:
            inf_y = inf_layer_height(f_d,V_out_max,rho,mew_kin,y_tar)
        
    if (inf_y > .1*D): # still too large
        # going to ramp up velocity until it is in a viable range
        while (inf_y > .1*D):
            if V_max == 'inlet':
                V = V*1.1
                Re = calc_Re(D,V,rho,mew) 
                f_d = calc_f_d(Re,Eps)
                inf_y = inf_layer_height(f_d,V,rho,mew_kin,y_tar)
            else:
                V_out_max = V_out_max*1.1
                Re_out = calc_Re(D_min,V_out_max,rho,mew)
                f_d = calc_f_d(Re_out,Eps)
                inf_y = inf_layer_height(f_d,V_out_max,rho,mew_kin,y_tar)
                # update the inlet velocity and reynolds #
                A_out_min = D_to_A(D_min)
                n_out = float(len(D_out))
                V = V_from_V_out_max(V_out_max,A_out_min,A_in,n_out)
                Re = calc_Re(D,V,rho,mew)

    return inf_y, V, Re



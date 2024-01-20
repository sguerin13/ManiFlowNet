# library imports
import os
from scripts.helpers import load_config

import sim.generation.y_plus_calcs as YPC 
import sim.generation.ansys_utils as AU
from sim.generation.fluent import FluentCommands as FC
from sim.generation.meshing import MeshCommands as MC
from sim.generation.simulation import SimulationCommands as SC
import time

if __name__ == "__main__":
    config = load_config(os.path.join("scripts", "sim", "generation","sim_routine.json"))

    ############################ Path and Defines ############################

    geo_dir = config.geometryDir
    flow_param_path = os.path.join("outputs", "flow_params.csv")
    save_file_path = config.simOutputPath
    
    # this is passed to ansys workbench so it can grab
    # the python files and read them as strings
    path_to_code = os.path.join(os.getcwd(),"sim", "generation")

    # Defines
    start_time = time.time()
    interactive = True
    n_iter = config.nIterations # set number of iterations to run in simulation
    msh_qual = config.meshQuality # target mesh cell quality
    msh_skew = config.meshSkew # target mesh skew value
    msh_n_cores = config.meshNCores # number of cores to use while meshing
    sim_n_cores = config.simNCores # number of cores to use in fluent sim

    sc = SC() # initialize simulation commands object

    # grab list of flow parameters
    Re_l, Eps_l, Visc_l, Rho_l, P_l = sc.grab_flow_params(flow_param_path)

    # create new folder and define the file_paths
    sim_no = str(len(os.listdir(save_file_path)))
    sim_path = os.path.join(save_file_path,sim_no)

    if not os.path.exists(sim_path):
        os.mkdir(sim_path)

    D_path = sim_path + '\\'
    log_path = os.path.join(sim_path,'log.txt')
    sim_status_path = os.path.join(sim_path,'sim_status.txt')
    mesh_exp_path = os.path.join(sim_path,'CFD_mesh.msh')
    res_path = os.path.join(sim_path,'res_file.txt')
    sol_path = os.path.join(sim_path,'sol_file.txt')


    try:
        
    ##################### SETUP ######################

    #Setup New Simulation System
        Reset()
        SetScriptVersion(Version="22.2.192")
        template1 = GetTemplate(TemplateName="Fluid Flow")
        system1 = template1.CreateSystem()
        sc.set_sys(system1)

        # create the new simulation status file
        AU.create_simulation_status_file(sim_status_path)

        ####################################################

        ################### GEOMETRY & BC ##################
        area_ratio_okay = False
        while not area_ratio_okay:

        # define directory to find the CAD Files:
            cad_file = sc.grab_cad_file(geo_dir)

            # load the geometry:
            sc.load_geo(cad_file)

            # grab pipe diameters
            D_in, D_out = sc.grab_pipe_diameter(D_path,path_to_code,interactive)

            # only inlet diameters, could make an argument for using all diameters
            D = sum(D_in)/float(len(D_in))
            # grab inlet and outlet area
            A_in, A_out = AU.grab_inlet_outlet_area(cad_file)
            if A_in/A_out > .25 and A_in/A_out < 5.0:
                area_ratio_okay = True
                #TODO: Add routine to flag file as too extreme an area_ratio
                

        # create the log file
        title_str = cad_file.split('Geometries\\')[1] + '\n'
        AU.create_log_file(log_path,title_str)

        #Grab Flow Parameters
        Re,Eps,visc,rho,P,visc_kin = sc.choose_flow_params(Re_l,
                                                            Eps_l, 
                                                            Visc_l, 
                                                            Rho_l, 
                                                            P_l)
            
        # determine pipe velocity,
        # impose velocity limits based on inlet-outlet areas and diameters
        # compate inlet velocity to maximum outlet velocity, use the largest 
        # to size the boundary layer
        V, V_out_max, Re = YPC.calc_V(Re,rho,visc,D_in,D_out,A_in,A_out)

        # determine the inflation layer size
        BL, V, Re = YPC.size_boundary_layer(D,
                                            Re,
                                            Eps,
                                            V,
                                            V_out_max,
                                            rho,
                                            visc,
                                            visc_kin,
                                            D_in,
                                            D_out,
                                            A_in,
                                            A_out)

        # calculate the Turbulent Inlet Intensity
        T_int = AU.calc_turb_int(Re)

        # write flow params to the log file:
        flow_params_log = AU.flow_params_str(Re   = Re,
                                            Eps  = Eps,
                                            Visc = visc,
                                            Rho  = rho,
                                            P    = P,
                                            V    = V,
                                            D    = D,
                                            A_in = A_in,
                                            A_out = A_out,
                                            T_int = T_int,
                                            ) 

        AU.update_log_file(log_path,flow_params_log)

        ##################################################

        ################## MESHING #######################
        # Run the Mesher
        meshComponent1 = system1.GetComponent(Name="Mesh")
        meshComponent1.Refresh()

        Mesh = sc.sys.GetContainer(ComponentName="Mesh")
        mesh_dict = {
                    'mesh_object': Mesh,
                    'D': D,
                    'BL': BL,
                    'qual': msh_qual,
                    'skew': msh_skew,
                    'log_path': log_path,
                    'sim_status_path': sim_status_path,
                    'AU_path': path_to_code,
                    'msh_path' : mesh_exp_path,
                    'n_cores' : msh_n_cores,
                    'interactive' : interactive
                    }

        mc = MC(mesh_dict)
        mc.create_mesh()
            #####################################################

        #     ##################### FLUENT SIM ####################

        #Run Fluent Sim
        setupComponent1 = system1.GetComponent(Name="Setup")
        setupComponent1.Refresh()
        setup1 = system1.GetContainer(ComponentName="Setup")
        print("Setting Up Fluid Sim")
        sim_dict = {
                    'Re': Re,
                    'Eps': Eps,
                    'Visc': visc,
                    'Rho': rho,
                    'P': P,
                    'V': V,
                    'D': D,
                    'conv_crit': 0.0001,
                    'T_Int': T_int,
                    'n_iter': n_iter,
                    'res_path': res_path,
                    'sol_path': sol_path,
                    'Sim_Object': setup1,
                    'n_cores': sim_n_cores,
                    'interactive' : interactive
                    }

        # run the simulation
        fc = FC(sim_dict)
        fc.setup_and_run()

        # log sim time
        end_time = time.time()
        run_time = "Run Time: " + str(end_time - start_time) + ","
        AU.update_log_file(log_path,run_time)

    except Exception as e:
        os.rename(sim_path,sim_path + '_failed')
        print("Simulation Failed: ",e)
        time.sleep(10)
'''
Class to execute all of the commands sent to Fluent

'''
import logging
import re

class FluentCommands():

    def __init__(self,sim_dict):
        self.sim_dict = sim_dict
        self.Re = self.sim_dict['Re']
        self.sim_object = self.sim_dict['Sim_Object']
        self.interactive = self.sim_dict['interactive']

    def setup_sim(self):
        '''
        setup the simulation
        '''
        self.fluent_launcher()
        self.open()
        self.set_fluid_props()
        self.choose_fluid()
        self.define_model()
        self.set_velo_BC()
        self.set_pressure_BC()
        self.set_wall_roughness()
        self.set_convergence_criteria()
        self.residuals_to_save()
        return

    def run_sim(self):
        '''
        run the simulation
        '''
        self.initialize()
        self.run_solver()
        self.save_residuals()
        self.step()
        self.save_solution()
        self.close()
        return
    
    def setup_and_run(self):
        '''
        do everything
        '''
        self.setup_sim()
        self.run_sim()
        return

    def fluent_launcher(self):

        fluentLauncherSettings1 = self.sim_object.GetFluentLauncherSettings()
        fluentLauncherSettings1.SetEntityProperties(Properties=Set(EnvPath={}, 
                                                    RunParallel=True))
        self.sim_object.GetFluentLauncherSettings().ShowLauncher = False # Showing Fluent
        self.sim_object.GetFluentLauncherSettings().Precision = 'Double' 
        self.sim_object.GetFluentLauncherSettings().RunParallel = True 
        self.sim_object.GetFluentLauncherSettings().NumberOfProcessors = \
            self.sim_dict['n_cores'] 
        return

    def open(self):
        self.sim_object.Edit(Interactive = self.interactive)
        return
    
    def close(self):
        self.sim_object.Exit()
        return
        

    def set_fluid_props(self):
        '''
        command line text to change fluid properties

        set density and viscosity
        '''

        fluid_prop_str = "define/materials/change-create air " + \
                         "liquid yes constant " + str(self.sim_dict['Rho']) + \
                         " no no yes constant " + str(self.sim_dict['Visc']) + \
                         " no no no;"
        
        self.sim_object.SendCommand(fluid_prop_str)
        return

    def choose_fluid(self):
        # setting the liquid to fluid and then using the rest of the defaults
        fluid_str = 'define/boundary-conditions/ fluid * () yes liquid' + \
        ' no no no no 0 no 0 no 0 no 0 no 0 no 1 no no no no no '
        self.sim_object.SendCommand(fluid_str)
        return


    def define_model(self):
        '''
        set the physics model:
        - K-Omega for turbulent flow
        - ______ for transitionary flow #TODO: update this
        - ______ for laminar flow

        '''
        if self.Re > 3000: # turbulent flow
            model_str = 'define/models/viscous/kw-sst? yes'
            model = 'turb'
        
        elif self.Re > 2000: # transitionary flow
            model_str = 'define/models/viscous/transition-sst? yes'
            model = 'trans'

        elif self.Re < 2000: # laminar flow
            model_str = 'define/models/viscous/laminar? yes'
            model = 'lam'

        self.sim_object.SendCommand(model_str)


        if model == 'trans':
            geo_wall_rough_str = 'define/models/viscous/' + \
            'trans-sst-roughness-correlation? yes no ' + \
            str(self.sim_dict['D']*self.sim_dict['Eps'])
            
            self.sim_object.SendCommand(geo_wall_rough_str)

        return

    def set_velo_BC(self):
        '''
        velocity inlet BC
        - Turbulent: also set turbulent inlet intensity, D is based on hydraulic
        diameter and will have to be different than D if using a non circular 
        manifold

        - Transition: need to set turbulent intensity and intermittence

        - Laminar: No turbulent intensity specified

        '''
        if self.Re > 3000:
            V_str = 'define/boundary-conditions/velocity-inlet inlet' + \
                    ' no no yes yes no ' + str(self.sim_dict['V']) + \
                    ' no 0 no no no yes ' + str(self.sim_dict['T_Int']) + \
                    ' ' + str(self.sim_dict['D']) + ' '

        elif self.Re > 2000:
            # Turbulent Specification Method: Intermittency, Intensity and Hydraulic Diameter
            V_str = 'define/boundary-conditions/velocity-inlet inlet' + \
                    ' no no yes yes no ' + str(self.sim_dict['V']) + \
                    ' no 0 no no no yes no 1 ' + str(self.sim_dict['T_Int']) + \
                    ' ' + str(self.sim_dict['D']) + ' '

        elif self.Re < 2000: 
            V_str = 'define/boundary-conditions/velocity-inlet inlet' + \
                    ' no no yes yes no ' + str(self.sim_dict['V']) + \
                    ' no 0 '

        self.sim_object.SendCommand(V_str)
        return

    def set_pressure_BC(self):
        '''
        Pressure Outlet BC
        - Turbulent: also set turbulent inlet intensity, D is based on hydraulic
        diameter and will have to be different than D if using a non circular 
        manifold

        - Transition:

        - Laminar: 

        '''
        if self.Re > 3000:
            P_str = 'define/boundary-conditions/pressure-outlet ' + \
                    'outlet yes no ' + str(self.sim_dict['P']) + \
                    ' no yes no no no yes ' + str(self.sim_dict['T_Int']) + \
                    ' ' + str(self.sim_dict['D']) + ' yes no yes no '

        elif self.Re > 2000:
            P_str = 'define/boundary-conditions/pressure-outlet ' + \
                    'outlet yes no ' + str(self.sim_dict['P']) + \
                    ' no yes no no no yes no 1 ' + str(self.sim_dict['T_Int']) + \
                    ' ' + str(self.sim_dict['D']) + ' yes no yes no '

        elif self.Re < 2000:
            P_str = 'define/boundary-conditions/pressure-outlet ' + \
                    'outlet yes no ' + str(self.sim_dict['P']) + \
                    ' no yes yes no yes no '
        
        self.sim_object.SendCommand(P_str)
        return

    def set_wall_roughness(self):
        '''
        # i believe this is based on D_h????
        # don't set wall roughness for laminar flow

        '''
        if self.Re > 2000:
            rgh_ht = self.sim_dict['D']*self.sim_dict['Eps']
            wall_str = 'define/boundary-conditions/wall body no no no no' + \
                    ' ' + str(rgh_ht) + ' no 0.5 '
            self.sim_object.SendCommand(wall_str)
        
        return


    def set_convergence_criteria(self):
        '''
        set the convergence criteria for the simulation


        '''
        if self.Re > 3000:
            # set convergence criteria in the monitor  # cont., x-velo, y-velo, z-velo, k-resid, omega
            conv_crit = str(self.sim_dict['conv_crit'])
            conv_str = 'solve/monitor/residual/convergence-criteria ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' '

        elif self.Re > 2000:
            # set convergence criteria in the monitor  
            # cont., x-velo, y-velo, z-velo, k-resid, omega, intermit, retheta
            conv_crit = str(self.sim_dict['conv_crit'])
            conv_str = 'solve/monitor/residual/convergence-criteria ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' '

        elif self.Re < 2000:
            # cont., x-velo, y-velo, z-velo,
            conv_crit = str(self.sim_dict['conv_crit'])
            conv_str = 'solve/monitor/residual/convergence-criteria ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' ' + \
                        conv_crit + ' '

        self.sim_object.SendCommand(conv_str)
        return

    def residuals_to_save(self):
        '''

        state the number of iterations that you want to save the residual
        data for

        '''
        iter_str = str(self.sim_dict['n_iter'])
        resid_str = '/solve/monitors/residual/n-save ' + iter_str + ' '
        self.sim_object.SendCommand(resid_str)
        return
    
    def save_residuals(self):
        # residuals file path
        res_path = self.sim_dict['res_path']
        resid_data_path = '/plot/residuals-set plot-to-file ' + res_path + ' '
        self.sim_object.SendCommand(resid_data_path)
        return

    def step(self):
        # residuals will not save w/o iterating 1 more time, appears to be a bug
        # with iteration count, so you have iterate 1 more time to save the residuals to file
        self.sim_object.SendCommand('it 1')
        return


    def initialize(self):
        self.sim_object.SendCommand('solve/initialize/hyb-initialization')
        return

    def run_solver(self):
        # define # of iterations and let it run - after the first 500 steps start checking the convergence
        # every 100 steps, if it isn't converging, cancel the sim and move onto the next one
        solver_str =  'solve/iterate ' + str(self.sim_dict['n_iter'])
        self.sim_object.SendCommand(solver_str)


        '''
        Code for evaluating convergence, currently not used
        '''
        # init_steps = 500
        # step_size = 100
        # step_count = 0
        # while step_count < self.sim_dict['n_iter']:
        #     if step_count == 0:
        #         solver_str =  'solve/iterate ' + str(init_steps)
        #         self.sim_object.SendCommand(solver_str)
        #         self.save_residuals()
        #         self.step()

        #         res_dict = self.read_res_file()
        #         sim_status = self.evaluate_convergence(res_dict)
        #         if sim_status == 'converged':
        #             break
        #         elif sim_status == 'converging':
        #             pass
        #         elif sim_status == 'stagnated':
        #             raise

        #         step_count += init_steps
        #         if step_count >= self.sim_dict['n_iter']:
        #             break
        #     else:
        #         solver_str =  'solve/iterate ' + str(step_size)
        #         self.sim_object.SendCommand(solver_str)
        #         self.save_residuals()
        #         self.step()
                
        #         res_dict = self.read_res_file()
        #         sim_status = self.evaluate_convergence(res_dict)
        #         if sim_status == 'converged':
        #             break
        #         elif sim_status == 'converging':
        #             pass
        #         elif sim_status == 'stagnated':
        #             raise

        #         step_count += step_size
        #         if step_count >= self.sim_dict['n_iter']:
        #             break
        return

    def read_res_file(self):

        residual_dict = {}
        with open(self.sim_dict['res_path'],'r') as f:
            lines = f.readlines()
            
            counter = 0
            # first pass, find all of the entry bounds
            for line in lines:

                # start of a region
                if re.search("^\(",line):
                    value = re.search("^\(\(.*",line).group(0).split(" ")[1]
                    value = value.split("\"")[1]
                    # print(value)
                    if value not in residual_dict.keys():
                        residual_dict[value] = {}

                if re.search("^\)",line):
                    # scientific notation
                    last_res = re.search("^[0-9]*\t[0-9]*\.[0-9]*e\-[0-9]*",lines[counter-1])
                    if last_res != None:
                        last_res = last_res.group(0)
                        step_num = last_res.split("\t")[0]
                        last_res = last_res.split("\t")[1]
                        residual_dict[value][int(step_num)] = float(last_res)

                    else: 
                        # decimal notation
                        last_res = re.search("^[0-9]*\t[0-9]*\.[0-9]*",lines[counter-1]).group(0)
                        step_num = last_res.split("\t")[0]
                        last_res = last_res.split("\t")[1]
                        residual_dict[value][int(step_num)] = float(last_res)
                    
                counter += 1
    
        return residual_dict


    def evaluate_convergence(self,residual_dict):
        continuity_keys = list(residual_dict['continuity'].keys())
        last_key = continuity_keys[-1]
        last_val = residual_dict['continuity'][last_key]
        if last_val <= self.sim_dict['conv_crit']:
            return 'converged'
        
        second_to_last_key = continuity_keys[-2]
        second_to_last_val = residual_dict['continuity'][second_to_last_key]
        delta = (last_val - second_to_last_val)/second_to_last_val
        if delta > -.1:
            return 'stagnated'
        else:
            return 'converging'


    def save_solution(self):
        #save data         no surfaces = (), yes commas, params 
        sol_path = self.sim_dict['sol_path']
        solution_file_str = '/file/export ascii ' + \
                            sol_path + ' () yes ' + \
                            'z-velocity y-velocity x-velocity pressure'
        self.sim_object.SendCommand(solution_file_str)
        return

import math
from sim.generation.ansys_utils import read_D,read_flow_params
import sim.generation.y_plus_calcs as YPC
import random
import os
import logging


class SimulationCommands():

    def __init__(self):
        pass

    def set_sys(self,sys):
        self.sys = sys

    def grab_cad_file(self, GeoDir):
        cad_files = os.listdir(GeoDir)
        # grad a random cad file
        cad_file  = random.choice(cad_files)
        cad_file = os.path.join(GeoDir,cad_file)
        return cad_file

    def load_geo(self,cad_file):
        self.geometry = self.sys.GetContainer(ComponentName="Geometry")
        self.geometry.SetFile(FilePath=cad_file)
        return

    def grab_pipe_diameter(self,pipe_d_path,script_path,interactive):
        
        # file path to store pipe diameter
        D_file_str  = 'D_path = ' + 'r\'' + pipe_d_path + '\\\'' + ' \n'

        # load the script file
        D_script_path = os.path.join(script_path,'pipe_diameter.py')
        script_file = open(D_script_path)
        script_str = script_file.read()
        script_file.close()

        D_script = D_file_str + script_str
        # run script to access pipe diameter from model
        # Mesh = self.sys.GetContainer(ComponentName="Mesh")
        # meshComponent1 = self.sys.GetComponent(Name="Mesh")
        # meshComponent1.Refresh()
        print("Opening Geometry")
        logging.info("Opening Geometry")
        self.geometry.Edit(IsSpaceClaimGeometry=True,Interactive = interactive) 
        self.geometry.SendCommand(Command=D_script,Language="Python")
        self.geometry.Exit()
        print("Closing Geometry")
        logging.info("Closing Geometry")
        d_path = os.path.join(pipe_d_path,'D_file.txt')
        D_in, D_out = read_D(d_path)
        return D_in, D_out

    def grab_flow_params(self,flow_param_path):

        flow_params = read_flow_params(flow_param_path)

        Re_list       = flow_params[0]
        Eps_list      = flow_params[1]
        Visc_list     = flow_params[2]
        Density_list  = flow_params[3]
        P_list        = flow_params[4]

        return Re_list, Eps_list, Visc_list, Density_list, P_list

    def choose_flow_params(self, Re_list, Eps_list,
                           Visc_list, Density_list, P_list):

        Re = random.choice(Re_list)
        Eps = random.choice(Eps_list)
        visc = random.choice(Visc_list)
        rho = random.choice(Density_list)
        P = random.choice(P_list)
        visc_kin = visc/rho

        return Re,Eps,visc,rho,P,visc_kin

    def choose_water_flow_params(self, Re_list, P_list, Eps_list):
        Re = random.choice(Re_list)
        P = random.choice(P_list)
        Eps = random.choice(Eps_list)
        rho = 997 # kg/m^3
        visc = 0.00089 # Pa - s
        visc_kin = visc/rho
        return Re,Eps,visc,rho,P,visc_kin


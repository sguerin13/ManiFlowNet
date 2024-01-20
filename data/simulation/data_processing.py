

import open3d as o3d
import pandas as pd
import pickle
from pyntcloud import PyntCloud as PC
import numpy as np
import data.simulation.ansys_msh_parser as AMP
import os
import re
import scipy

class SimulationDataProcessing():

    def __init__(self,sim_path,cfd_mesh_file,mesh_dict_file,
                solution_file,point_cloud_csv,point_cloud_file,
                bc_file,log_file,fluid_prop_file):

        self.sim_path = sim_path
        self.solution_file = os.path.join(sim_path,solution_file)
        self.cfd_mesh_file = os.path.join(sim_path,cfd_mesh_file)
        self.mesh_dict_file = os.path.join(sim_path,mesh_dict_file)
        self.point_cloud_csv = os.path.join(sim_path,point_cloud_csv)
        self.point_cloud_file = os.path.join(sim_path,point_cloud_file)
        self.bc_file = os.path.join(sim_path,bc_file)
        self.log_file = os.path.join(sim_path,log_file)
        self.fluid_prop_file = os.path.join(sim_path,fluid_prop_file)

    ##################### MAIN ROUTINE ####################################

    def create_data_structures(self):
        self.create_volume_mesh()
        self.create_pnt_cloud_csv()
        self.fuse_data()
        self.process_null_nodes()
        self.create_pc_ply(ascii = True)
        self.pull_bcs()
        self.pull_fluid_props()

    #######################################################################


    ##################### CREATE 3D Datastructures ########################
    def create_pnt_cloud_csv(self):

        '''
        rearranges the csv layout so Pyntcloud can create a PLY file

        '''
        pandas_list = pd.read_csv(self.solution_file)
        cl = list(pandas_list.columns)
        cl = list([cl[1],cl[2],cl[3],cl[4],cl[5],cl[6],cl[7],cl[0]])
        pandas_list = pandas_list.reindex(columns = cl)
        pandas_list.columns.values[0] = 'x'
        pandas_list.columns.values[1] = 'y'
        pandas_list.columns.values[2] = 'z'
        pandas_list.to_csv(self.point_cloud_csv,index=False)
        return

    def create_pc_ply(self,ascii=True):
        # creates a point cloud .ply file from a csv that
        # was reordered by the create_pnt_cloud_csv function

        pnt_cld = PC.from_file(self.point_cloud_csv)
        pnt_cld.to_file(self.point_cloud_file,as_text = ascii)
        return

    def create_volume_mesh(self):
        AMP.create_and_save_mesh_dict(self.cfd_mesh_file,self.mesh_dict_file)
        return

    def create_surface_mesh(mesh_dict):
        pass

    def fuse_data(self):
        ''' share data between files and update both data structures
            - Point cloud shares physical node values with the mesh dict
            - Mesh shares zone assignments with the point cloud csv
        '''
        # load files
        mesh_dict = pickle.load(open(self.mesh_dict_file,'rb'))
        df        = pd.read_csv(self.point_cloud_csv)
        
        n_points = mesh_dict['points']['num_points']
        
        # add new column to the dataframe, pre_load w/ ones
        zones = [1]*n_points
        mesh_node_nums = [-1]*n_points
        df['zone'] = zones
        df['mesh_node_number'] = mesh_node_nums

        # match entries and share info
        for i in range(1,n_points+1):
            # if i%100 == 0:
            #     print(i)
            msh = mesh_dict['points']['point_data'][i]
            msh_xyz = msh['xyz']
            df_index = df[np.isclose(df['x'],msh_xyz[0]) & 
                        np.isclose(df['y'],msh_xyz[1]) & 
                        np.isclose(df['z'],msh_xyz[2])].index.tolist()
            assert len(df_index)==1 

            # add the mesh node number
            df.iloc[df_index,9] = i

            # update dictionary
            msh['pressure']   = df.iloc[df_index,3].tolist()[0]
            msh['x_velocity'] = df.iloc[df_index,4].tolist()[0]
            msh['y_velocity'] = df.iloc[df_index,5].tolist()[0]
            msh['z_velocity'] = df.iloc[df_index,6].tolist()[0]

            # update dataframe
            # multizone node - assign to inlet or outlet via max
            if len(msh['zone_id']) > 1:
                if 7 in msh['zone_id']:
                    df.iloc[df_index[0],8] = 7 # wall

                elif 5 in msh['zone_id']: 
                    df.iloc[df_index[0],8] = 5 # inlet
                
                elif 6 in msh['zone_id']:
                    df.iloc[df_index[0],8] = 6 # outlet
                
                else:
                    raise ValueError('Incorrect Zone Id\'s in mesh dict')

            else:
                df.iloc[df_index[0],8] = msh['zone_id']
        
        assert -1 not in df['mesh_node_number'].tolist()
            
        # update files
        pickle.dump(mesh_dict, open(self.mesh_dict_file, "wb" ) )
        df.to_csv(self.point_cloud_csv,index=False)
        return

    def process_null_nodes(self,mode='interpolate'):
        '''
        some node values contain all zero's, not sure why, but to address that
        we will take the neighboring nodes and average their values and use that
        to fill in the missing data. 

        Once the mesh_dict is updated, we will then update those points in the dataframe
        
        '''
        mesh_dict = pickle.load(open(self.mesh_dict_file,'rb'))
        df        = pd.read_csv(self.point_cloud_csv)
        n_points = mesh_dict['points']['num_points']
        
        for i in range(1,n_points+1):
            node = mesh_dict['points']['point_data'][i]
            node_x_v = node['x_velocity']
            node_y_v = node['y_velocity']
            node_z_v = node['z_velocity']
            node_p   = node['pressure']
            node_values = [node_x_v,node_y_v,node_z_v,node_p]

            if any(node_values) == False: # node has all zeros
                # print(i)
                x_v, y_v, z_v, p = [],[],[],[]
                neighbor_ids = node['connected_nodes']

                for node_id in neighbor_ids:
                    neigh_node = mesh_dict['points']['point_data'][node_id]
                    neigh_x_v = neigh_node['x_velocity']
                    neigh_y_v = neigh_node['y_velocity']
                    neigh_z_v = neigh_node['z_velocity']
                    neigh_p   = neigh_node['pressure']
                    neigh_node_values = [neigh_x_v,neigh_y_v,neigh_z_v,neigh_p]
                    
                    # make sure this neighbor node isn't also a null node
                    if any(neigh_node_values) == True:
                        x_v.extend([neigh_x_v])
                        y_v.extend([neigh_y_v])
                        z_v.extend([neigh_z_v])
                        p.extend([neigh_p])

                # update the node value with new interpolated values
                node['x_velocity'] = np.mean(x_v)
                node['y_velocity'] = np.mean(y_v)
                node['z_velocity'] = np.mean(z_v)
                node['pressure']   = np.mean(p)
                mesh_dict['points']['point_data'][i] = node

                # update the point clouds
                msh_xyz = node['xyz']
                df_index = df[np.isclose(df['x'],msh_xyz[0]) & 
                              np.isclose(df['y'],msh_xyz[1]) & 
                              np.isclose(df['z'],msh_xyz[2])].index.tolist()
                assert len(df_index)==1, "node has duplicate at node: %d" % id

                # update point cloud
                df.iloc[df_index,3] = node['pressure']
                df.iloc[df_index,4] = node['x_velocity']
                df.iloc[df_index,5] = node['y_velocity']
                df.iloc[df_index,6] = node['z_velocity']

        # save the files
        pickle.dump(mesh_dict, open(self.mesh_dict_file, "wb" ))
        df.to_csv(self.point_cloud_csv,index=False)

    def pull_bcs(self):
        # pull the velocity and the pressure measures from the pnt_cld_file
        pnt_cld = PC.from_file(self.point_cloud_csv)
        vx = pnt_cld.points[pnt_cld.points['zone']==5].iloc[0,4] # 5
        vy = pnt_cld.points[pnt_cld.points['zone']==5].iloc[0,5] # 5
        vz = pnt_cld.points[pnt_cld.points['zone']==5].iloc[0,6] # 5
        v_mag = np.linalg.norm([vx,vy,vz],2)
        p  = pnt_cld.points[pnt_cld.points['zone']==6].iloc[0,3] # 6
        
        out_tup = (v_mag,p)
        pickle.dump(out_tup, open(self.bc_file, "wb" ) )
        return

    def pull_fluid_props(self):

        with open(self.log_file,'r') as f:
            lines = f.readlines()
            
            for i in range(len(lines)):
                txt = lines[i]
                # file name
                if i == 0:
                    pass

                # flow params
                if i == 1:
                    flow_params_dict = {}
                    Re = re.search("Re\: [0-9]*\.[0-9]*",txt).group(0)
                    flow_params_dict['Re'] = float(Re.split(" ")[1])

                    Eps = re.search("Eps\: [0-9]*\.[0-9]*",txt).group(0)
                    flow_params_dict['Eps'] = float(Eps.split(" ")[1])

                    Visc = re.search("Visc\: [0-9]*\.[0-9]*",txt).group(0)
                    flow_params_dict['Visc'] = float(Visc.split(" ")[1])

                    Rho = re.search("Rho\: [0-9]*\.[0-9]*",txt).group(0)
                    flow_params_dict['Rho'] = float(Rho.split(" ")[1])

                    P = re.search("P\: [0-9]*\.[0-9]*",txt).group(0)
                    flow_params_dict['P'] = float(P.split(" ")[1])
                    
                    V = re.search("V\: [0-9]*\.[0-9]*",txt).group(0)
                    flow_params_dict['V'] = float(V.split(" ")[1])

                    D = re.search("D\: [0-9]*\.[0-9]*",txt).group(0)
                    flow_params_dict['D'] = float(D.split(" ")[1])

        pickle.dump(flow_params_dict, open(self.fluid_prop_file, "wb" ))
        return            

    @staticmethod
    def build_adjacency_list(mesh_file):
        mesh_dict = pickle.load(open(mesh_file,'rb'))
        n_points = mesh_dict['points']['num_points']
        node = mesh_dict['points']['point_data']
        adj_list = []
        for i in range(1,n_points+1): # points are 1-indexed
            adj_list.extend([list(node[i]['connected_nodes'])])
        
        return adj_list

    @staticmethod
    def build_adjacency_list_from_dict(mesh_dict):
        nodes = mesh_dict['points']['point_data']
        adj_list = []
        for i in nodes.keys(): # points are 1-indexed
            adj_list.extend([(i,j) for j in nodes[i]['connected_nodes']])
        return adj_list

    @staticmethod
    def build_adjacency_matrix(mesh_file):
        mesh_dict = pickle.load(open(mesh_file,'rb'))
        n_points = mesh_dict['points']['num_points']
        node = mesh_dict['points']['point_data']
        adj_mat = np.zeros((n_points,n_points))
        for i in range(1,n_points+1): # points are 1-indexed
            for j in node[i]['connected_nodes']:
                adj_mat[i-1,j-1]=1
        
        return adj_mat

    ###############################################################################
        
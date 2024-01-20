'''
Ansys Mechanical Mesher Object

Methods create inflation layer, set meshing parameters and update the mesh


'''
import logging

class MeshCommands():
    
    def __init__(self,mesh_dict):
        self.mesh_dict = mesh_dict
        self.mesh_obj = self.mesh_dict['mesh_object']
        self.AU_path = self.mesh_dict['AU_path']
        self.log_path = self.mesh_dict['log_path']
        self.sim_status_path = self.mesh_dict['sim_status_path']
        self.interactive = self.mesh_dict['interactive']


    ### routines ###
    def create_mesh(self):
        self.open()
        self.add_inflation_layer()
        self.set_mesh_attributes()
        logging.info("Updating Mesh")
        self.update_mesh()
        self.check_cell_limit()
        
        self.log_mesh_metrics(self.AU_path,self.log_path)
        self.export_mesh()
        logging.info("Closing Mesher")
        self.close()

    def set_mesh_attributes(self):
        self.set_elem_size(size = self.mesh_dict['D']/float(10))
        self.set_msh_qual_tgt(qual = self.mesh_dict['qual'])
        self.set_smoothing()
        self.set_curve_angle()
        self.set_msh_qual_tgt(qual = self.mesh_dict['qual']) # was this added on purpose? TODO
        self.set_msh_skew_tgt(skew = self.mesh_dict['skew'])
        self.set_num_cpu_cores(n_cores = self.mesh_dict['n_cores'])
        return

    ### sub routines ###

    def update_mesh(self):
        # update the mesh
        update_str = \
        'model  = ExtAPI.DataModel.Project.Model \n' + \
        'msh    = model.Mesh \n' + \
        'msh.Update() \n'
        self.mesh_obj.SendCommand(Command = update_str,Language = 'Python')
        return

    def open(self):
        self.mesh_obj.Edit(Interactive = self.interactive)
        return

    def close(self):
        self.mesh_obj.Exit()
        return

    def add_inflation_layer(self,max_layers = 3):

        # create the inflation layer
        inf_str = \
        'model  = ExtAPI.DataModel.Project.Model \n' + \
        'msh    = model.Mesh \n' + \
        'body = ExtAPI.DataModel.GeoData.Assemblies[0].Parts[0].Bodies[0] \n' + \
        'ns     = model.NamedSelections \n' + \
        'inlet  = ns.Children[0] \n' + \
        'outlet = ns.Children[1] \n' + \
        'body_ns = ns.Children[2] \n' + \
        'bod_ids = [] \n' + \
        'bod_ids.Add(body.Id) \n' + \
        'geo_selection  = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities) \n' + \
        'geo_selection.Ids = bod_ids \n' + \
        'inflation = ExtAPI.DataModel.Project.Model.Mesh.AddInflation() \n' + \
        'inflation.PropertyByName("GeometryDefineBy").InternalValue= 0 \n' + \
        'inflation.Location = geo_selection \n' + \
        'inflation.PropertyByName("BoundaryDefineBy").InternalValue= 1 \n' + \
        'inflation.BoundaryLocation = body_ns \n'

        # sizing the boundary layer
        BL = self.mesh_dict['BL']
        BL = str(BL)
        BL_str = \
        'BL = ' + BL + '\n' + \
        'BL_str = str(BL) + \' [m]\' \n' + \
        'inflation.InflationOption = 1 \n' + \
        'inflation.MaximumLayers = ' + str(max_layers) + ' \n' + \
        'inflation.FirstLayerHeight = Quantity(BL_str) \n'

        inf_str = inf_str + BL_str

        self.mesh_obj.SendCommand(Command = inf_str,Language = 'Python')
        return
    

    def set_elem_size(self,size):
        size_str = str(size)
        elem_size_str = \
        'msh  = ExtAPI.DataModel.Project.Model.Mesh \n' + \
        'elem_size = ' + size_str + ' \n' + \
        'elem_size_str = str(elem_size) + \' [m]\' \n' + \
        'msh.ElementSize = Quantity(elem_size_str) \n'
        self.mesh_obj.SendCommand(Command = elem_size_str,Language = 'Python')
        return

    def set_smoothing(self,n = 2):
        # smoothing of cell growth: 0-low,1-default,2-high
        msh = 'msh  = ExtAPI.DataModel.Project.Model.Mesh \n'
        smoothing_str = msh + 'msh.Smoothing = ' + str(n) + ' \n'
        self.mesh_obj.SendCommand(Command = smoothing_str,Language = 'Python')
        return

    def set_curve_angle(self,ang = 12):
        msh = 'msh  = ExtAPI.DataModel.Project.Model.Mesh \n'
        curv_str = msh + \
        'msh.CurvatureNormalAngle = Quantity(\'' + str(ang) + ' [deg]\') \n'
        self.mesh_obj.SendCommand(Command = curv_str,Language = 'Python')
        return

    def set_msh_qual_tgt(self,qual = 0.7):
        msh = 'msh  = ExtAPI.DataModel.Project.Model.Mesh \n'
        qual_str = msh + \
        'msh.TargetQuality = ' + str(qual) + '\n'

        self.mesh_obj.SendCommand(Command = qual_str,Language = 'Python')
        return
        
    def set_msh_skew_tgt(self,skew = .4):
        msh = 'msh  = ExtAPI.DataModel.Project.Model.Mesh \n'
        skew_str = msh + \
        'msh.TargetSkewness = Quantity(\'' + str(skew) + ' [m]\') \n'
        self.mesh_obj.SendCommand(Command = skew_str,Language = 'Python')
        return

    def set_num_cpu_cores(self,n_cores):
        n_core_str = \
        "ExtAPI.DataModel.Project.Model.Mesh.InternalObject.NumCpuPartMeshing = " \
        + str(n_cores)
        self.mesh_obj.SendCommand(Command = n_core_str,Language = 'Python')

    def check_cell_limit(self):
        '''

        check the element count to make sure it is less than 500K
        if over 500K increase the mesh cell size by 10% iteratively until
        under 500K cells, if at any point meshing fails, move onto the next iteration

        '''
        msh = 'msh  = ExtAPI.DataModel.Project.Model.Mesh \n'
        cell_limit_str = msh + \
        'if msh.Elements > 5e5: \n' + \
        '\t while msh.Elements > 5e5: \n' + \
        '\t\t elem_size = float(msh.ElementSize.ToString().split(\' \')[0]) \n' + \
        '\t\t elem_size = elem_size*1.1 \n' + \
        '\t\t elem_size_str = str(elem_size) + \' [m]\' \n' + \
        '\t\t msh.ElementSize = Quantity(elem_size_str) \n' + \
        '\t\t msh.Update() \n'
        # '\t\t if msh.InternalObject.MeshingFailed: \n' + \
        # '\t\t\t break'
        # print("Checking Cell Limit")
        self.mesh_obj.SendCommand(Command = cell_limit_str,Language = 'Python')
                
    def log_mesh_metrics(self,AU_path,log_path):
        # log the mesh quality metrics
        
        import_str = \
        'import sys \n' + \
        'sys.path.append(' + '\'' + AU_path + '\\\'' + ') \n' + \
        'import ansys_utils as AU \n' 
        log_str = 'log_path = ' + '\'' + log_path + '\' \n'
        msh = 'msh  = ExtAPI.DataModel.Project.Model.Mesh \n'

        log_str = log_str + import_str + msh + \
        'msh.MeshMetric = 1 \n' + \
        'min_q = msh.PropertyByName("MeshMetricMin").InternalValue \n' + \
        'max_q = msh.PropertyByName("MeshMetricMax").InternalValue \n' + \
        'av_q = msh.PropertyByName("MeshMetricAverage").InternalValue \n' + \
        'std_q = msh.PropertyByName("MeshMetricSTDV").InternalValue \n' + \
        'qual_str = AU.mesh_qual_metrics_str(min_q, max_q, av_q, std_q) \n'
        log_str = log_str + 'AU.update_log_file(log_path,qual_str) \n'
        self.mesh_obj.SendCommand(Command = log_str,Language = 'Python')

    def check_mesh_fail(self,AU_path,sim_status_path,log_path):
        ''' If the mesh has failed, set simulation status to 0 so that the 
        simulation routine throws an error and moves onto the next iteration '''
        print("generating mesh failure check string")
        import_str = \
        'import sys \n' + \
        'sys.path.append(' + '\'' + AU_path + '\\\'' + ') \n' + \
        'import ansys_utils as AU \n' 
        sim_status_str = 'sim_status_path = ' + '\'' + sim_status_path + '\' \n'
        log_str = 'log_path = ' + '\'' + log_path + '\' \n'
        msh = 'msh  = ExtAPI.DataModel.Project.Model.Mesh \n'

        check_status_str = import_str + sim_status_str + log_str + msh + \
        'if msh.InternalObject.MeshingFailed: \n' + \
        '\t AU.set_simulation_status_fail(sim_status_path) \n' + \
        '\t log_str = \'Sim Failure: Meshing Failed,\' \n' + \
        '\t AU.update_log_file(log_path,log_str) \n'
        print("sending command")
        self.mesh_obj.SendCommand(Command = check_status_str,Language = 'Python')
        return

    def check_simulation_status(self,fpath):
        with open(fpath,'r') as f:
            status = f.read()
        return bool(int(status))
    
    def export_mesh(self):

        exp_str = 'exp_path = ' + '\'' + self.mesh_dict['msh_path'] + '\' \n' +\
        'ExtAPI.DataModel.Project.Model.Mesh.InternalObject.WriteFluentInputFile(exp_path) \n'
        self.mesh_obj.SendCommand(Command = exp_str,Language = 'Python')

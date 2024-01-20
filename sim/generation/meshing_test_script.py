
import ansys_utils as AU

'''
Test script for the meshing routine

'''



# text that will be added to this script
'D = D_Val'
'BL = BL_Val'
'log_path = path_str'

model  = ExtAPI.DataModel.Project.Model
msh    = model.Mesh
ns     = model.NamedSelections
inlet  = ns.Children[0]
outlet = ns.Children[1]
body_ns= ns.Children[2]

# assign the part body
body = ExtAPI.DataModel.GeoData.Assemblies[0].Parts[0].Bodies[0]
bod_ids = []
bod_ids.Add(body.Id)
geo_selection  = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
geo_selection.Ids = bod_ids

# geometry scope
# add the inflation layer
inflation = ExtAPI.DataModel.Project.Model.Mesh.AddInflation()
inflation.PropertyByName("GeometryDefineBy").InternalValue= 0
inflation.Location = geo_selection

# named selections
inflation.PropertyByName("BoundaryDefineBy").InternalValue= 1
inflation.BoundaryLocation = body_ns


# set inflation layer type to 'first layer thickness'
# BL = .0001
BL_str = str(BL) + ' [m]'

inflation.InflationOption = 1
inflation.MaximumLayers = 3
inflation.FirstLayerHeight = Quantity(BL_str)


#mesh attributes
elem_size = D/float(10)
elem_size_str = str(elem_size) + ' [m]'
msh.ElementSize = Quantity(elem_size_str)
msh.Smoothing = 2 # high
msh.CurvatureNormalAngle = Quantity("12 [deg]")
msh.TargetQuality = Quantity(".7 [m]")
msh.TargetSkewness = Quantity(".4 [m]")

# update the mesh
msh.Update()


#check the element count to make sure it is less than 500K
if msh.Elements > 5e5:
    while msh.Elements > 5e5:
        elem_size = elem_size*1.1 
        elem_size_str = str(elem_size) + ' [m]'
        msh.ElementSize = Quantity(elem_size_str)
        msh.Update()


# log the mesh quality metrics
msh.MeshMetric = 1
min_q = msh.PropertyByName("MeshMetricMin").InternalValue
max_q = msh.PropertyByName("MeshMetricMax").InternalValue
av_q = msh.PropertyByName("MeshMetricAverage").InternalValue
std_q = msh.PropertyByName("MeshMetricSTDV").InternalValue

qual_str = AU.mesh_qual_metrics_str(min_q, max_q, av_q, std_q)
AU.update_log_file(log_path,qual_str)

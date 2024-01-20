import math

# This is a snippet of code that is passed in as a string to ansys workbench
# don't need to worry about the variables that are referenced here without 
# being defined, they are defined in a different section in MeshCommands

inlet_list = Selection.CreateByGroups("inlet").Items
outlet_list = Selection.CreateByGroups("outlet").Items

body = GetRootPart().Bodies[0]
inlet_diameters = []
outlet_diameters = []

for face in body.Faces:
    if inlet_list.Contains(face) == True:
        D = math.sqrt(face.Area*4.0/math.pi)
        inlet_diameters.extend([D])
    
    if outlet_list.Contains(face) == True:
        D = math.sqrt(face.Area*4.0/math.pi)
        outlet_diameters.extend([D])


with open(D_path + 'D_file.txt', "w") as f:
    f.write('Inlet Diameters:')
    for i in inlet_diameters:
        f.write(str(i) + ',')
    f.write('\n')
    f.write('Outlet Diameters:')
    for i in outlet_diameters:
        f.write(str(i) + ',')
    

    
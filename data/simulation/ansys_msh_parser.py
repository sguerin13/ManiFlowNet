import numpy as np
import re
import pickle


'''
# copying structure laid out in meshio w/ some edits

Want to include data about it's BC and adjacency, which is not included
in the base meshio processing library

regex overview:
https://docs.python.org/3/howto/regex.html

[a-d] # specify character class, '-' gives a range


########### High Level Overview of Functions ############

find_data_inds():
	- Figure out where the sections of information start and end

read_ind_desc():
	
	-  Read the index descriptions '(# (0 ...))' to get information
	    about the number of lines included for each section

	- Also initializes the mesh dictionary to store data


read_points():
	- Reads point data from the text file
    - assigns xyz location to the point in the dictionary


read_cells():
	- Reads cell data


read_faces():
	- Reads faces and uses helper functions to update connectivity 
    data in the mesh dictionary
    - Also assigns a zone id to the node

read_zones():
	- Function that reads the zone id's in the mesh dictionary and 
    updates the zone name field in the dictionary

Consolidate_point_data():
	- Removes duplicate connections and zone_id data in the dictionary 
    by removing duplicates

read():
	- Reads the file and returns the mesh dictionary
    - calls most of the above functions in sequence

create_and_save_mesh_dict():
	- Summary function runs all other functions and creates the mesh 
    dictionary for a given file and saves the dictionary. 

##### Helper Functions 

_get_nodes():
	- Helper function that reads the type of faces and then grabs 
    the nodes accordingly

_get_connectivity():
	- Helper function that generates the connectivity based on the 
    number of nodes on the face, takes information from the _get_nodes function
	
    - Returns a list with sublists that include connectivity data

_update_dict_connect_data
	- Updates the connectivity entry in the mesh dictionary for each node. 

_zone_id_lookup():
	- Receives zone id and return zone string 


'''

def find_data_inds(file_path):
    data_start_ind = []
    data_end_ind   = []
    with open(file_path,"rb") as f:
        iterator=1
        while True:
            line=f.readline().decode('utf-8')
            if not line:
                break
            else:
                '''
                Looking for the beginning of a section containing information
                match any amount of spaces, then a '(' and then match any amount 
                of spaces followed by a group of numbers () [0-9] as long as 
                there are one or more '+' and then any number of characters 
                after '.'

                '''
                out = re.match("\s*\(\s*([0-9]+).*", line)
                if out is not None:
                    data_start_ind.extend([iterator])


                out2 = re.match(".*([)])",line)
                if out2 is not None:
                    data_end_ind.extend([iterator])

                iterator+=1
    return data_start_ind,data_end_ind


def read_index_desc(st,file_path):
    '''
    Read the index descriptions '(# (0 ...))' to get information
    about the number of lines included for each section

    '''
    mesh_dict = {}
    with open(file_path,"rb") as f:
        iterator=1
        while True:
            line=f.readline().decode('utf-8')
            if not line:
                break

            elif iterator in st:
                out = re.match("\s*\(\s*([0-9]+).*", line)
                index = out.group(1)

                if index == '0':
                    # header file, do nothing
                    pass

                elif index == '2':
                    # dimensionality info, do nothing
                    pass

                elif index == '10' and line.count("(") == line.count(")"):
                    # this line describes the point information
                    out = re.match("\\s*\\(\\s*(10)\\s*\\(([^\\)]*)\\).*", line)
                    info = [int(num,16) for num in out.group(2).split()]
                    mesh_dict['points']={}
                    mesh_dict['points']['num_points'] = info[2]

                elif index == '12' and line.count("(") == line.count(")"):
                    # information about the cells
                    out = re.match("\\s*\\(\\s*(12)\\s*\\(([^\\)]*)\\).*", line)
                    info = [int(num,16) for num in out.group(2).split()]
                    mesh_dict['cells']={}
                    mesh_dict['cells']['num_cells'] = info[2]

                elif index == '13' and line.count("(") == line.count(")"):
                    # information about the Faces
                    out = re.match("\\s*\\(\\s*(13)\\s*\\(([^\\)]*)\\).*", line)
                    info = [int(num,16) for num in out.group(2).split()]
                    mesh_dict['faces']={}
                    mesh_dict['faces']['num_faces'] = info[2]
                
            iterator +=1
        # create space for storing point, face, and cell data
        mesh_dict['points']['point_data'] = {}
        mesh_dict['cells']['cell_data']   = {}
        mesh_dict['faces']['face_data']  = {}
    return mesh_dict


def read_points(st,mesh_dict,file_path):
    '''
    (point_index (zone_id, first_index, last_index, type, ND)
    (10 (2 1 4053 1 3)( ....

    point_index = 10 tells you that the following info is re: points
    zone_id     = tells you what part of the mesh each node belongs to

    first_ind   = index number of first point, usually 1
    last_ind    = index of last point
    type        = 1 if any type of node, 2 if it is a boundary node
    ND          = number of dimensions... equal to 3


    create a list entry for each point read that includes:
    [zone_id, index, (x,y,z coords), [connected nodes]]

    '''
    with open(file_path,"rb") as f:
        iterator=1
        while True:
            line=f.readline().decode('utf-8')
            
            if not line:
                break

            elif iterator in st:
                out = re.match("\s*\(\s*([0-9]+).*", line)
                index = out.group(1)

                if index == '10' and line.count("(") != line.count(")"):
                    '''
                    If we are reading point data, iterate over the lines of text
                    that stores node locations, keeping track of index 

                    '''
                    out = re.match("\\s*\\(\\s*(10)\\s*\\(([^\\)]*)\\).*", line)
                    info = [int(num,16) for num in out.group(2).split()]
                    zone_id, start_ind, end_ind = info[0], info[1], info[2]
                    iterator +=1 

                    for i in range(start_ind,end_ind+1):
                        line = f.readline().decode('utf-8')
                        coords = line.split()
                        xyz_tup  = (float(coords[0]),
                                    float(coords[1]),
                                    float(coords[2]))
                        mesh_dict['points']['point_data'][i] = {}
                        # mesh_dict['points']['point_data'][i]['zone_id'] = zone_id
                        mesh_dict['points']['point_data'][i]['zone_id'] = []
                        mesh_dict['points']['point_data'][i]['zone_name'] = []
                        mesh_dict['points']['point_data'][i]['xyz'] = xyz_tup
                        mesh_dict['points']['point_data'][i]['connected_nodes'] = []
                        iterator +=1

                    continue # skip iterator at the end since iter in for - loop 
            
            iterator +=1
    
    return mesh_dict

def read_cells(st,mesh_dict,file_path):
    '''
    Read the line at the end of the msh file that indicates the cell type:
        - 1: Triangle
        - 2: tetrahedral
        - 3: quadralateral
        - 4: hexahedral
        - 5: pyramid
        - 6: wedge

    zone header: (12(zone_id,first_ind,last_ind,type, element-type))
    zone-id = 4 indicates fluid body
    type = 1 means it is an active zone
    element-type = 0 indicates mixed elements

    element type per cell listed in line (2 2 2 2 .....)
    '''
    with open(file_path,"rb") as f:
        iterator=1
        while True:
            line=f.readline().decode('utf-8')
            
            if not line:
                break

            elif iterator in st:
                out = re.match("\s*\(\s*([0-9]+).*", line)
                index = out.group(1)

                if index == '12' and line.count("(") != line.count(")"):
                    
                    out = re.match("\\s*\\(\\s*(12)\\s*\\(([^\\)]*)\\).*", line)
                    info = [int(num,16) for num in out.group(2).split()]
                    start_ind, end_ind, elem_type = info[1], info[2], info[-1]
                    iterator +=1
                    line = f.readline().decode('utf-8')
                    cell_type = line.split()

                    mesh_dict['cells']['cell_data'] = \
                    {i:{} for i in range(1,len(cell_type)+1)}
                    
                    for i in range(1,len(cell_type)+1):
                        mesh_dict['cells']['cell_data'][i]['type'] \
                        = int(cell_type[i-1])

                        if mesh_dict['cells']['cell_data'][i]['type'] == 1:
                            mesh_dict['cells']['cell_data'][i]['type_name'] = \
                            'triangle'

                        elif mesh_dict['cells']['cell_data'][i]['type'] == 2:
                            mesh_dict['cells']['cell_data'][i]['type_name'] = \
                            'tetrahedral'
                        
                        elif mesh_dict['cells']['cell_data'][i]['type'] == 3:
                            mesh_dict['cells']['cell_data'][i]['type_name'] = \
                            'quadralateral'

                        elif mesh_dict['cells']['cell_data'][i]['type'] == 4:
                            mesh_dict['cells']['cell_data'][i]['type_name'] = \
                            'hexahedral'

                        elif mesh_dict['cells']['cell_data'][i]['type'] == 5:
                            mesh_dict['cells']['cell_data'][i]['type_name'] = \
                            'pyramid'

                        elif mesh_dict['cells']['cell_data'][i]['type'] == 6:
                            mesh_dict['cells']['cell_data'][i]['type_name'] = \
                            'wedge'

            iterator +=1

    return mesh_dict

def read_faces(st,mesh_dict,file_path):
    '''
    (13 (zone_id, first_ind, last_ind, type, elem_type))

    zone_id: the zone that the face belongs to

    type: boundary condition type: interior, wall, etc

    elem_type: type of face (line, triangle, square, etc...
               if 0, first entry of each row indicates element type
               else, use it to adjust what you are reading. 
    
    each row contains (elem_type) node 0, node_1,..., face_1,face_2

    Only care about node connectivity, going to ignore connected faces,
    per gambit file format, nodes are ordered ccw

    3-------2
    |       |
    |       |
    0-------1

    '''
    with open(file_path,"rb") as f:
        iterator=1
        while True:
            line=f.readline().decode('utf-8')
            
            if not line:
                break


            elif iterator in st:
                out = re.match("\s*\(\s*([0-9]+).*", line)
                index = out.group(1)

                if index == '13' and line.count("(") != line.count(")"):
                    out = re.match("\\s*\\(\\s*(13)\\s*\\(([^\\)]*)\\).*", line)
                    info = [int(num,16) for num in out.group(2).split()]
                    zone_id,elem_type = info[0],info[-1]
                    start_ind, end_ind, elem_type = info[1], info[2], info[-1]
                    iterator +=1
                    
                    for i in range(start_ind,end_ind+1):
                        line = f.readline().decode('utf-8')
                        line_data = [int(num,16) for num in line.split()]
                        n_nodes,connect_nodes = _get_nodes(elem_type,line_data)
                        tup_list = _get_connectivity(n_nodes,connect_nodes)
                        # print(tup_list)
                        mesh_dict = _update_dict_connect_data(mesh_dict,
                                                           zone_id,
                                                           tup_list)
                        iterator +=1 

                    continue

            iterator +=1
    return mesh_dict

def read_zones(mesh_dict):
    ''' read zone id's and write zone names to the dictionary entry'''

    n_points = mesh_dict['points']['num_points']
    for i in range(1, n_points+1):
        zone_list = mesh_dict['points']['point_data'][i]['zone_id']
        for elem in zone_list:
            mesh_dict['points']['point_data'][i]['zone_name'].extend([ 
                                                           _zone_id_lookup(elem)
                                                                     ])
    return mesh_dict

def consolidate_point_data(mesh_dict):
    '''
     consolidate mesh and zone_id data in the node dictionary items
     by removing duplicates
    
    '''
    n_points = mesh_dict['points']['num_points']
    for i in range(1, n_points+1):
        zone_list = mesh_dict['points']['point_data'][i]['zone_id']
        mesh_dict['points']['point_data'][i]['zone_id'] = np.unique(zone_list)
        c_nodes = mesh_dict['points']['point_data'][i]['connected_nodes']
        mesh_dict['points']['point_data'][i]['connected_nodes'] = \
            np.unique(c_nodes)

    return mesh_dict

def read(file_path):

    st,end = find_data_inds(file_path) #find the start and end of chunks of data
    mesh_dict = read_index_desc(st,file_path) # find out how pnts, cells, etc..
    mesh_dict = read_points(st,mesh_dict,file_path) #read the point cells
    mesh_dict = read_cells(st,mesh_dict,file_path)
    mesh_dict = read_faces(st,mesh_dict,file_path)
    mesh_dict = consolidate_point_data(mesh_dict)
    mesh_dict = read_zones(mesh_dict)

    return st,mesh_dict

def create_and_save_mesh_dict(file_path,save_file_name):

    st,mesh_dict = read(file_path)
    pickle.dump( mesh_dict, open( save_file_name, "wb" ) )

############ HELPER FUNCTIONS ############


def _get_nodes(elem_type,line_data):
    ''' read the face element type data and grabs nodes accordingly '''
    if elem_type == 0:
        face_type = line_data[0]

        if face_type == 2:
            # only 2 nodes on the face
            # node_zero = line_data[1]
            connect_nodes = line_data[1:3]
            n_nodes = 2

        elif face_type == 3:
            # 3 nodes on the face
            # node_zero = line_data[1]
            connect_nodes = line_data[1:4]
            n_nodes = 3
            
        elif face_type == 4:
            # 4 nodes on the face
            # node_zero = line_data[1]
            connect_nodes = line_data[1:5]
            n_nodes = 4

    elif elem_type == 2:
        # node_zero = line_data[0]
        connect_nodes = line_data[0:2]
        n_nodes = 2

    elif elem_type == 3:
        # node_zero = line_data[0]
        connect_nodes = line_data[0:3]
        n_nodes = 3
    
    elif elem_type == 4:
        # node_zero = line_data[0]
        connect_nodes = line_data[0:4]
        n_nodes = 4

    return n_nodes,connect_nodes #node_zero,

def _get_connectivity(n_nodes, con_nodes):
    ''' 
    - return connectivity data of the nodes 
    - Build a tuple for each node in the list, first node is node of interest
    other nodes in the tuple are connected to the first node


    2 nodes:   0 ------- 1
    3 nodes:         2
                    / \
                   /   \
                  0-----1
    4 nodes
                 3-------2
                 |       |
                 |       |
                 0-------1
            
    '''
    connect_list = []
    if n_nodes == 2:
        for i in range(0,n_nodes):
            node_list = [con_nodes[i], [con_nodes[i-1]] ]  # [(0,1),(1,0)]
            connect_list.extend([node_list])

    elif n_nodes == 3:
        for i in range(0,n_nodes):
            if i == (n_nodes - 1):  # [(2-->[0,1])]
                node_list = [con_nodes[i], [con_nodes[i-1],con_nodes[0]] ]
            
            else:                   # [(0-->[1,2]), (1-->[0,2])]
                node_list = [con_nodes[i], [con_nodes[i-1],con_nodes[i+1]] ]
            
            connect_list.extend([node_list])

    elif n_nodes == 4:
        for i in range(0,n_nodes):
            if i == (n_nodes - 1):
                node_list = [con_nodes[i], [con_nodes[i-1],con_nodes[0]] ]
            
            else:
                node_list = [con_nodes[i], [con_nodes[i-1],con_nodes[i+1]] ]
            
            connect_list.extend([node_list])

    return connect_list

def _update_dict_connect_data(mesh_dict,zone_id,node_list):
    '''
    Update the connectivity data of each node from the data gathered from
    the face entries of the mesh file
    '''

    for elem in node_list:
        mesh_dict['points']['point_data'][elem[0]]['zone_id'].extend([zone_id])
        mesh_dict['points']['point_data'][elem[0]]['connected_nodes'].extend(
                                                                      elem[1])

    return mesh_dict
        
def _zone_id_lookup(zone_id):
    ''' provide zone id number and returns a string with the zone name
    
    (45 (1 interior interior-straight_manifold_9_a_in_0.01641_a_out_0.05231_solid)())
    (45 (2 fluid straight_manifold_9_a_in_0.01641_a_out_0.05231_solid)())
    (45 (5 velocity-inlet inlet)())
    (45 (6 pressure-outlet outlet)())
    (45 (7 wall body)())
    
    '''
    if zone_id == 1:
        zone = 'interior'
    
    elif zone_id ==4:
        zone = 'fluid_body'
    
    elif zone_id ==5:
        zone = 'inlet'
    
    elif zone_id ==6:
        zone = 'outlet'
    
    elif zone_id ==7:
        zone = 'wall'
    
    else:
        raise ValueError('Zone Id not Valid:',zone_id)
    
    return zone




####################################################################


########## SANITY CHECK FUNCTIONS ##################################

def check_multi_zone_nodes(mesh_dict):
    multi_zone_nodes = []
    zone_list = []
    n_points = mesh_dict['points']['num_points']
    for i in range(1, n_points+1):
        z_list = mesh_dict['points']['point_data'][i]['zone_id']
        zone_list.extend([np.unique(z_list)])
        if len(np.unique(zone_list[-1])) > 1:
            multi_zone_nodes.extend([i])
    
    return zone_list,multi_zone_nodes



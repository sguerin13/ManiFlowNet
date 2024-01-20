import math
import random as R
import time
import sys

'''
Open SpaceClaim and run this script in the code terminal

Set the save_dir to the directory where you want to save the generated geometries
'''


save_dir = "Path/To/Geometries/SimpleManifold/"
n_geo_to_build = 250 # set the number of geometries to generate


class SimpleManifold():

    '''

    '''

    def __init__(self,pipe_dict):
    
        self.pd = pipe_dict
        self.pipe_d_var = self.pd['pipe_d_var']
        self.pipe_d_min = self.pd['pipe_d_min']
        self.seg_len_var= self.pd['seg_len_var']
        self.seg_len_min = self.pd['seg_len_min']
        self.n_outlet_var = self.pd['n_outlet_var']
        self.n_outlet_min = self.pd['n_outlet_min']
        self.outlet_d_var = self.pd['outlet_d_var']
        self.outlet_d_min = self.pd['outlet_d_min']
        self.outlet_len_var = self.pd['outlet_len_var']
        self.outlet_len_min = self.pd['outlet_len_min']
        
        self.in_circle_d = R.random()*self.pipe_d_var + self.pipe_d_min
        self.out_circle_d = R.random()*self.in_circle_d*self.outlet_d_var + \
                            self.in_circle_d*self.outlet_d_min

        self.outlet_len = R.random()*self.in_circle_d*self.outlet_len_var + \
                        self.in_circle_d*self.outlet_len_min

        self.origin = Point.Create(M(0.0), M(0.0), M(0.0))
        
        # member variables
        self.pipe_len = 0
        self.pipe_main_seg = []
        self.outlet_region_min = 0
        self.outlet_region_max = 0
        self.outlet_list = []
        self.num_out = 0
        self.outlet_pt = []

        self.curve_dict = {}
        self.hole_dict = {}
        self.outlet_line_points = []
        self.inlet_dir = None

        self.dist_bet_out = 0
        self.out_dir_list = []
        self.branch_curve_list = []
        self.branch_curve_names = []
        self.inlet_area = 0
        self.outlet_area = 0



    ############# Main Functions ##############

    def build_pipe(self):
        '''
        main geometry construction routine

        '''

        self.create_main_manifold_points()
        self.create_outlet_lines()
        self.create_geometry()
        self.check_single_body()
        if self.clean_part == True:
            self.name_all_faces()
        else:
            self.delete_part()

    def save_pipe_as_geo_file(self):
        self.calculate_in_out_area([self.origin],self.outlet_pt)
        self.save_file()

    ############# Geometry Creation #############

    def create_main_manifold_points(self):

        '''
        - creates the main pipe segment, stores it as a list of points

        '''

        pipe_len_var = self.pd['seg_len_var']
        pipe_len_min = self.pd['seg_len_min']
        self.pipe_len = (R.random()*pipe_len_var + \
                        pipe_len_min)*self.in_circle_d
        pipe_dir = [1,0,0]
        self.inlet_dir = Direction.Create(pipe_dir[0],pipe_dir[1],pipe_dir[2])
        self.pipe_main_seg = self.create_line_seg_list( self.origin,
                                                        self.pipe_len,
                                                        pipe_dir)

    def create_outlet_lines(self):

        '''
        - Creates the list of outlet line segments, each segment is a list
          of 2 points

        - Also checks that outlet size is within an acceptable range

        - Logs the outlet points which is used for naming faces

        '''

        self.num_out = R.randint(1,self.pd['n_outlet_var']) + \
                  self.pd['n_outlet_min']

         # region where we will allow outlets to exists 
        self.outlet_region_min = self.pipe_len*.1
        self.outlet_region_max = self.pipe_len*.9
        outlet_len = self.outlet_region_max - self.outlet_region_min

        outlet_locations = []
        for i in range(1,self.num_out+1):
            out_loc = outlet_len*i/(self.num_out+1) + self.outlet_region_min
            outlet_locations.extend([out_loc])
            print(out_loc)

        # check that the outlet diameters don't intersect and won't go outside
        # of the outlet region
        self.check_outlet_sizing(outlet_len)

        # create the outlet segments
        self.outlet_segments = []
        for i in range(self.num_out):
            outlet_anchor_pt = Point.Create(M(outlet_locations[i]),M(0),M(0))
            outlet_dir = [0,1,0]
            outlet_seg = self.create_line_seg_list(outlet_anchor_pt,
                                                    self.outlet_len,
                                                    outlet_dir)
            self.outlet_segments.extend([outlet_seg])


        # take inventory of all outlet parts
        self.outlet_pt.extend([self.pipe_main_seg[1]])
        for i in range(self.num_out):
            self.outlet_pt.extend([self.outlet_segments[i][1]])
        
        

    def create_geometry(self):

        '''
        - Creates the geometries by lofting each line segment for main seg and
          outlets

        '''

        # loft the main pipe
        p1 = self.pipe_main_seg[0]
        p2 = self.pipe_main_seg[1]
        self.loft_line_seg(p1, p2, self.in_circle_d/2.0, self.in_circle_d/2.0)
        
        for i in range(self.num_out):
            p1 = self.outlet_segments[i][0]
            p2 = self.outlet_segments[i][1]
            self.loft_line_seg(p1,p2,self.out_circle_d/2.0,self.out_circle_d/2.0)


    ############### HELPERS ###################

    def create_line_seg_list(self,origin,seg_len,direction):
        '''
        Creates a list of 2 points based on origin and a direction + length

        '''
        seg_list = []
        seg_list.append(origin)
        seg_list.append(Point.Create(origin.X + M(seg_len)*direction[0],
                                     origin.Y + M(seg_len)*direction[1],
                                     origin.Z + M(seg_len)*direction[2]))
        return seg_list 

    def check_outlet_sizing(self,outlet_len):
        '''
        
        - shrink outlet diameter by 10 % until the satisfactory conditions 
          are met: no intersection, and doesn't go outside of region
        
        '''
        while ((self.out_circle_d >= outlet_len/(self.num_out+1)) or 
              (outlet_len/(self.num_out+1) < self.out_circle_d/2.0)): 

              self.out_circle_d = self.out_circle_d*.9

    def loft_line_seg(self,p1,p2,r_st,r_end):
        '''
        lofts a line segment which is a list of 2 points and 2 radii

        '''
        direction = Direction.Create(p2.X-p1.X, p2.Y-p1.Y, p2.Z-p1.Z)
        st_circle  = self.create_circle(p1, direction, r_st)
        end_circle = self.create_circle(p2, direction, r_end)
        self.loft_circle_surfaces([st_circle,end_circle])
        
    def create_circle(self,point, direction, radius):
        circle = CircularSurface.Create(radius, direction, point)
        return circle.CreatedBody #returns IDOCobject

    def loft_circle_surfaces(self,circle_list):
        sel = Selection.Create(circle_list)
        options = LoftOptions()
        options.ExtrudeType = ExtrudeType.ForceAdd
        options.GeometryCommandOptions = GeometryCommandOptions()
        result = Loft.Create(sel, None, options)
        



################### Face Naming and Geometry Checking ####################
    def name_faces(self,face_name,point_list):
        '''
        point_list: list of Point objects

        Name the faces that contain the points in a given list,
        this is used to name inlet and outlet faces

        '''
        face_list = []
        for face in GetRootPart().Bodies[0].Faces:
            for point in point_list:
                if face.Shape.ContainsPoint(point): # inlet
                    face_list.append(face)

        primarySelection = FaceSelection.Create(face_list)
        secondarySelection = Selection()
        result = NamedSelection.Create(primarySelection, secondarySelection)
        result = NamedSelection.Rename("Group1", face_name)
        return face_list

    def name_body_faces(self,inlet_faces,outlet_faces):
        '''
        inlet_faces: list of Face objects used for manifold inlet
        outlet_faces: list of Face objects used for manifold outlet

        - Use the list of inlet and outlet faces to name all other faces as the
          body

        '''
        body_face_list = []
        for face in GetRootPart().Bodies[0].Faces:
            if face not in inlet_faces and face not in outlet_faces:
                body_face_list.append(face)

        primarySelection = FaceSelection.Create(body_face_list)
        secondarySelection = Selection()
        result = NamedSelection.Create(primarySelection, secondarySelection)
        result = NamedSelection.Rename("Group1", "body")
        return body_face_list

    def name_all_faces(self):

        '''
        - routine to name the faces with either the single inlet and multiple 
        outlets or in reverse with multiple inlets and a single outlet

        '''

        in_faces = self.name_faces('inlet',[self.origin])
        out_faces = self.name_faces('outlet',self.outlet_pt)
        body_faces = self.name_body_faces(in_faces,out_faces)
        


    def clear_face_names(self):
        '''
        - Part of cleanup routine to remove face naming groups


        '''
        result  = NamedSelection.Delete('inlet')
        result  = NamedSelection.Delete('outlet')
        result  = NamedSelection.Delete('body')

    def calculate_in_out_area(self,input_pts,output_pts):

        '''
        Calculate the inlet and outlet surface areas, this will be used to 
        limit velocities during the simulation

        '''
        A_in = 0.0
        A_out = 0.0
        for face in GetRootPart().Bodies[0].Faces:
            for point in input_pts:      
                if face.Shape.ContainsPoint(point): # inlet
                    A_in += face.Area
            
            for point in output_pts:
                if face.Shape.ContainsPoint(point): # inlet
                    A_out += face.Area

        self.inlet_area = A_in
        self.outlet_area = A_out

    def save_file(self):
        '''
        File Naming Scheme:
        1_to_{num_out}_{face_mode}_A_in_{inlet_area}_A_out_{outlet_area}.scdoc
        
        '''
        save_file_str = 'straight_manifold_' + str(self.num_out) + '_A_in_' + \
                         "{:1.5f}".format(self.inlet_area) + '_A_out_' + \
                         "{:1.5f}".format(self.outlet_area) + '.scdoc'
        save_str = self.pd['save_dir'] + save_file_str
        options = ExportOptions.Create()
        DocumentSave.Execute(save_str, options)
        
    def check_single_body(self):
        '''
        Check to make sure we have a single part body if not remove the part

        '''
        n_body = GetRootPart().GetAllBodies()
        if len(n_body) > 1: # something went wrong
            self.clean_part = False
        else:
            self.clean_part = True

        if self.clean_part == False:
            self.delete_part()

    
    def delete_part(self):
        '''
        delete the created geometry

        '''
        for i in reversed(range(len(GetRootPart().GetAllBodies()))):
            selection = BodySelection.Create(GetRootPart().Bodies[i])
            result = Delete.Execute(selection)
        
        for i in reversed(range(len(GetRootPart().Curves))):
            selection = CurveSelection.Create(GetRootPart().Curves[i])
            result = Delete.Execute(selection)





##############   Implementation   ###############


pipe_dict = {

# pipe diameter
'pipe_d_var' : .190,
'pipe_d_min' : .010,

# length of the pipe as a multiple of the pipe diameter
'seg_len_var'   : 20.0,
'seg_len_min'   : 5.0,

# n outlets
'n_outlet_min' : 2,
'n_outlet_var' : 8,

# outlet diameter as a multiple of inlet diameter
'outlet_d_var' : .7,
'outlet_d_min' : .2,

# outlet length
'outlet_len_var' : 2.0,
'outlet_len_min' : 1.0,


'save_dir': save_dir

}

i = 0
while i < n_geo_to_build:
        
    pipe = SimpleManifold(pipe_dict)
    pipe.build_pipe()
    if pipe.clean_part == True:
        pipe.save_pipe_as_geo_file()
        i += 1
        pipe.delete_part()



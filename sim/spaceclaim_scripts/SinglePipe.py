# Python Script, API Version = V19 Beta
# Python Script, API Version = V19 Beta

import random as R


'''
Open SpaceClaim and run this script in the code terminal

Set the save_dir to the directory where you want to save the generated geometries
'''

save_dir = "Path/To/Geometries/SinglePipe/"
n_geo_to_build = 250 # set the number of geometries to generate



class SinglePipe():
    def __init__(self,pipe_dict):
        self.pd = pipe_dict
        self.n_points = self.pd['n_points']
        self.pipe_d_var = self.pd['pipe_d_var']
        self.pipe_d_min = self.pd['pipe_d_min']
        self.seg_len_var= self.pd['seg_len_var']
        self.seg_len_min = self.pd['seg_len_min']
        self.in_circle_d = R.random()*self.pipe_d_var + self.pipe_d_min
        self.out_circle_d = R.random()*self.pipe_d_var + self.pipe_d_min
        self.origin = Point.Create(M(0.0), M(0.0), M(0.0))
        self.face_mode = self.pd['face_mode']
        
        # member variables
        self.point_list = List[Point]()



    def build_pipe(self):
        self.create_curve()
        self.create_geometry()
        self.check_single_body()
        self.check_three_faces()
        if self.clean_part == True:
            self.name_all_faces()
        else:
            self.delete_part()

    
    def save_pipe_as_geo_file(self):
        self.calculate_in_out_area([self.origin],[self.outlet_pt])
        self.save_file()       

    ###### main geometry generation functions ####

    def create_curve(self):
        '''
        Create the curve the manifold

        - Define points that will create the manifold centerline
        - Define the inlet and outlet tangents
        - Create the Nurbs Curve
        
        '''
        # Create Points
        xyz_coords = \
        [self.seg_len_var*R.random()+self.seg_len_min for _ in range(3)]

        self.point_list.Add(self.origin)
        for i in range(1,self.n_points):
            if i == 1:
                second_point_x = self.seg_len_var*R.random()*self.in_circle_d + \
                    self.in_circle_d*self.seg_len_min
                xyz_coords = [second_point_x,0,0]
            else:
                xyz_coords = \
                [self.seg_len_var*R.random()*self.in_circle_d + \
                    self.in_circle_d*self.seg_len_min for _ in range(3)]

            self.point_list.Add(Point.Create(
                                M(self.point_list[i-1][0] + xyz_coords[0]),
                                M(self.point_list[i-1][1] + xyz_coords[1]),
                                M(self.point_list[i-1][2] + xyz_coords[2])))

        self.outlet_pt = self.point_list[self.n_points-1]

        # create tangents
        inlet_dir = [1,0,0]
        self.inlet_dir = Direction.Create(inlet_dir[0],inlet_dir[1],inlet_dir[2])
        
        # half the difference between x_0 and x_1
        in_tan_x_len = abs(self.point_list[0].X - self.point_list[1].X)
        self.inlet_tangent = \
        Vector.Create(self.inlet_dir.X*in_tan_x_len, 0, 0)

        out_dir_x = 0 
        out_dir_y = self.point_list[self.n_points-1].Y
        out_dir_z = self.point_list[self.n_points-1].Z
        self.outlet_dir =  Direction.Create(out_dir_x, out_dir_y, out_dir_z)
        self.outlet_tangent = \
        Vector.Create(out_dir_x, out_dir_y, out_dir_z)

        # create curve
        ncurve = NurbsCurve.CreateThroughPoints(False, self.point_list, 0.0001, 
                                        startDerivative = self.inlet_tangent,
                                        endDerivative = self.outlet_tangent)

        curveSegment = CurveSegment.Create(ncurve)
        designCurve = DesignCurve.Create(GetRootPart(),curveSegment)
        designCurve.SetName('centerline')
        self.center_line = designCurve
    
    def create_geometry(self):
        '''
        - Creates the geometries by populating the curves with circles which
          and then creates a loft from all of the circles on the curve 

        '''
        # inlet_pt = self.point_list[0]
        # inlet_circle = self.create_circle(inlet_pt,self.inlet_dir,
        #                                   self.inlet_circle_d/2)


        out_circle = self.create_circle(self.outlet_pt, self.outlet_dir, 
                                        self.out_circle_d/2)

        manifold_circles = [out_circle]
        manifold_circles = self.populate_curve_w_circles(n_div = 50,
                                    curve = self.center_line,
                                    st_circ = self.in_circle_d/2.0,
                                    end_circ = self.out_circle_d/2.0,
                                    circ_list = manifold_circles)

        self.loft_circle_surfaces(manifold_circles)

    def create_circle(self,point, direction, radius):
        circle = CircularSurface.Create(radius, direction, point)
        return circle.CreatedBody #returns IDOCobject

    def radius_interpolate(self,r_st,r_end,eval_pt):
        return (1.0-eval_pt)*r_st + eval_pt*r_end

    def loft_circle_surfaces(self,circle_list):
        sel = Selection.Create(circle_list)
        options = LoftOptions()
        options.ExtrudeType = ExtrudeType.ForceAdd
        options.GeometryCommandOptions = GeometryCommandOptions()
        result = Loft.Create(sel, None, options)

    def tangent_direction(self,incoming_curve, evaluation_point):
        '''
            incoming_curve: A Spaceclaim curve(Curve Object)
            evaluation_point: A number between 0-1(Float)
            - evaluates the tangent at the evaluation point along the curve

        '''

        interim_1 = incoming_curve.EvalProportion(evaluation_point) #Find a point on the curve
        bottom_point = interim_1.Point
        interim_2 = incoming_curve.EvalProportion(evaluation_point +.001)#Find another point on the curve that is extremely close to the first point
        top_point = interim_2.Point
        X1 = bottom_point.X - top_point.X #Find the tangent direction by finding the vector between the two very close points
        Y1 = bottom_point.Y - top_point.Y
        Z1 = bottom_point.Z - top_point.Z
        direction = Direction.Create(X1,Y1,Z1)
        return (bottom_point, direction)#bottom_point: point object ||| direction: direction object

    def populate_curve_w_circles(self,n_div,curve,st_circ,end_circ,circ_list):
        '''
        - populate curve with circles that interpolate from start radius to 
          end radius along the curve

        '''
        for i in range(n_div):
            eval_pt = (1.0-float(i+1)/float(n_div))
            pt_loc, pt_tan = self.tangent_direction(curve, eval_pt)
            pt_radii = self.radius_interpolate(st_circ, end_circ, eval_pt)
            circ_list.append(self.create_circle(pt_loc, pt_tan, pt_radii))
        
        return circ_list

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

        if self.face_mode == 'normal':
            in_faces = self.name_faces('inlet',[self.origin])
            out_faces = self.name_faces('outlet',[self.outlet_pt])
            body_faces = self.name_body_faces(in_faces,out_faces)
        
        elif self.face_mode == 'reverse':
            in_faces = self.name_faces('inlet',[self.outlet_pt])
            out_faces = self.name_faces('outlet',[self.origin])
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
        
        if self.face_mode == 'reverse':
            # swap the inlet and outlet areas
            A_in, A_out = A_out, A_in

        self.inlet_area = A_in
        self.outlet_area = A_out

    def save_file(self):
        '''
        
        File Naming Scheme:
        1_to_{num_out}_{face_mode}_A_in_{inlet_area}_A_out_{outlet_area}.scdoc
        
        '''
        save_file_str = 'single_manifold_' + \
                         self.face_mode + '_A_in_' + \
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


    def check_three_faces(self):
        if len(GetRootPart().Bodies[0].Faces) > 3:
            self.clean_part == False


    
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




pipe_dict = {

# n curve points
'n_points' : 5,

# pipe diameter
'pipe_d_var' : .190,
'pipe_d_min' : .010,

# plane that the outlet points will reside on
'seg_len_var'   : 3,
'seg_len_min'   : 1,

'face_mode': 'normal',
'save_dir': save_dir
}

i = 0
n_geo_to_build = 250
while i < n_geo_to_build:
        
    pipe = SinglePipe(pipe_dict)
    pipe.build_pipe()
    if pipe.clean_part == True:
        # save file, then reverse in-out orientation and save
        pipe.save_pipe_as_geo_file()
        pipe.clear_face_names() # reverse faces
        pipe.face_mode = 'reverse'
        pipe.name_all_faces()
        pipe.save_pipe_as_geo_file()
        i += 1
        pipe.delete_part()
        
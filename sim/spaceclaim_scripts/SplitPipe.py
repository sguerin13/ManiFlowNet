# Python Script, API Version = V19 Beta
# Python Script, API Version = V19 Beta

import random as R

'''

Open SpaceClaim and run this script in the code terminal

Set the save_dir to the directory where you want to save the generated geometries

'''


save_dir = "Path/To/Geometries/SplitPipe/"
n_geo_to_build = 250 # set the number of geometries to generate
class SplitPipe():

    def __init__(self,pipe_dict):
        
        self.pd = pipe_dict
        self.pipe_d_var = self.pd['pipe_d_var']
        self.pipe_d_min = self.pd['pipe_d_min']
        self.in_circle_d = R.random()*self.pipe_d_var + self.pipe_d_min
        self.out_circle_d = R.random()*self.pipe_d_var + self.pipe_d_min
        self.origin = Point.Create(M(0.0), M(0.0), M(0.0))
        self.pipe_mode = self.pd['pipe_mode']
        self.face_mode = self.pd['face_mode']
        
        # member variables
        self.curve_dict = {}
        self.hole_dict = {}
        self.outlet_line_points = []
        self.inlet_dir = None
        self.pipe_stem_len = 0
        self.pipe_stem_list = []
        self.num_out = 2
        self.dist_bet_out = 0
        self.outlet_list = []
        self.out_dir_list = []
        self.branch_curve_list = []
        self.branch_curve_names = []
        self.inlet_area = 0
        self.outlet_area = 0

    def build_pipe(self):
        self.create_stem()
        self.create_outlet_points()
        self.create_outlet_anchor_points()
        self.create_outlet_line_points()
        self.create_curves()
        self.create_geometry()
        self.check_single_body()
        if self.clean_part == True:
           self.name_all_faces()
        else:
           self.delete_part()
            

    def save_pipe_as_geo_file(self):
        self.calculate_in_out_area([self.origin],self.outlet_list)
        self.save_file()        

        
    
    ###################### Pipe Building Functions ##########################

    def create_stem(self):
        '''
        - creates the straight pipe inlet component

        '''
        
        pipe_stem_var = self.pd['pipe_stem_var']
        pipe_stem_min = self.pd['pipe_stem_min']
        self.pipe_stem_len = (R.random()*pipe_stem_var + \
                        pipe_stem_min)*self.in_circle_d/2.0
        stem_dir = [1,0,0]
        self.inlet_dir = Direction.Create(stem_dir[0],stem_dir[1],stem_dir[2])
        self.pipe_stem_list = self.create_line_seg_list(self.origin,
                                                       self.pipe_stem_len,
                                                       stem_dir
                                                       )

    def create_outlet_points(self):
        '''
        - Decide number of outlet points
        - Determine distance between outlet points
        - Determine location of first outlet point based on random scalar
            of inlet diameter for x,y,z locations
        - Create other outlet points offset from first based on dist_bet_out
        '''

        # define the outlet plane and first outlet point
        self.outlet_list = List[Point]()
        first_outlet = [self.pd['outlet_plane_var']*R.random()*self.in_circle_d+ \
                        self.pd['outlet_plane_min']*self.in_circle_d for _ in range(3)]
        
        first_outlet[0] = first_outlet[0] + self.pipe_stem_list[-1].X
        first_outlet[1] = first_outlet[1] + self.pipe_stem_list[-1].Y
        first_outlet[2] = first_outlet[2] + self.pipe_stem_list[-1].Z
        
        # add first point
        self.outlet_list.Add(Point.Create(M(first_outlet[0]),
                                          M(first_outlet[1]),
                                          M(first_outlet[2])))
        
        # add second point reflected along the xy plane
        self.outlet_list.Add(Point.Create(M(first_outlet[0]),
                                          M(first_outlet[1]),
                                          M(first_outlet[2]*-1)))

    def create_outlet_anchor_points(self):
        '''
        - Creates the anchor point for the outlet curves
        - These are offset from the inlet stem due to issues with 
          spaceclaim merging bodies where they share common faces

        '''
        self.anchor_points = []
        for i in range(len(self.outlet_list)):
            if i == 0:
                x = self.pipe_stem_list[-1].X + (self.outlet_list[i].X - self.pipe_stem_list[-1].X)/2.0
                y = 0
                z = 0
                self.anchor_points.append(Point.Create(x,y,z))
            else:
                x = self.anchor_points[i-1].X + self.dist_bet_out/2.0
                y = 0
                z = 0
                self.anchor_points.append(Point.Create(x,y,z))
        return



    def create_outlet_line_points(self):
        '''
        - Creates the intermediate point between the anchor point and the outlet 
          point
        - Depending on the 'mode' it will lock the intermediate point to the 
          y or z axis, otherwise it will be interpolated between the location
          of the anchor point and the outlet point
        '''

        for i in range(2):
            out_line = List[Point]()
            # out_line.Add(self.pipe_stem_list[0])
            out_line.Add(self.anchor_points[i])

            if self.pipe_mode == 'interpolate':
                int_pnt = Point.Create(self.anchor_points[i].X + 
                          (self.outlet_list[i].X - self.anchor_points[i].X)/1.33,
                           self.outlet_list[i].Y/3.0,
                           self.outlet_list[i].Z/3.0)

            out_line.Add(int_pnt)
            out_line.Add(self.outlet_list[i])
            self.outlet_line_points.append(out_line)

        # outlet
        for i, outlet in enumerate(self.outlet_list):
            out_dir_x = (outlet.X - self.anchor_points[i].X)
            print(out_dir_x,self.anchor_points[i].X,outlet.X)
            out_dir_y = outlet.Y
            out_dir_z = outlet.Z
            self.out_dir_list.append(Direction.Create(out_dir_x,
                                                      out_dir_y,
                                                      out_dir_z))


    def create_curves(self):
        '''
        - Create the curve for each of the outlet curves

        '''
        for i in range(2):
            branch = self.outlet_line_points[i]  # list of points
            branch_dir = self.out_dir_list[i]
            
            branch_start_tangent = Vector.Create(self.inlet_dir.X*self.in_circle_d*7.5,
                                                 self.inlet_dir.Y*self.in_circle_d*7.5,
                                                 self.inlet_dir.Z*self.in_circle_d*7.5)
            
            branch_end_tangent = Vector.Create(branch_dir.X*self.out_circle_d*10.0,
                                               branch_dir.Y*self.out_circle_d*10.0,
                                               branch_dir.Z*self.out_circle_d*10.0)
            
            ncurve = NurbsCurve.CreateThroughPoints(False, branch, 0.0001,
                                                    startDerivative = branch_start_tangent,
                                                    endDerivative = branch_end_tangent)

            curveSegment = CurveSegment.Create(ncurve)
            branch_designCurve = DesignCurve.Create(GetRootPart(),curveSegment)
            branch_designCurve.SetName('outlet'+str(i))
            self.branch_curve_list.append(branch_designCurve)
            
    
    def create_geometry(self):
        '''
        - Creates the geometries by populating the curves with circles which
          and then creates a loft from all of the circles on the curve 

        '''
        for i in range(2):
            if i != self.num_out - 1:
                # create outlet point
                outlet_pt = self.outlet_list[i]
                outlet_dir = self.out_dir_list[i]
                out_circle = self.create_circle(outlet_pt, outlet_dir, 
                                                self.out_circle_d/2)
                branch_circles = [out_circle]
                branch_circles = self.populate_curve_w_circles(n_div = 50,
                                            curve = self.branch_curve_list[i],
                                            st_circ = self.in_circle_d/2.1,
                                            end_circ = self.out_circle_d/2.0,
                                            circ_list = branch_circles)
                # stem_circle = self.create_circle(self.pipe_stem_list[0],
                #                             self.inlet_dir,
                #                             self.in_circle_d/2)
                # anchor_circle = self.create_circle(self.anchor_points[-2],
                #                             self.inlet_dir,
                #                             self.in_circle_d/2)                            
                # branch_circles.append(stem_circle)
                self.loft_circle_surfaces(branch_circles)
            else:
                outlet_pt = self.outlet_list[i]
                outlet_dir = self.out_dir_list[i]
                out_circle = self.create_circle(outlet_pt, outlet_dir, self.out_circle_d/2)
                branch_circles = [out_circle]
                branch_circles = self.populate_curve_w_circles(n_div = 30,
                                                            curve = self.branch_curve_list[i],
                                                            st_circ = self.in_circle_d/2.0,
                                                            end_circ = self.out_circle_d/2.0,
                                                            circ_list = branch_circles)
                # stem_circle = self.create_circle(self.pipe_stem_list[0],
                #                             self.inlet_dir,
                #                             self.in_circle_d/2)
                # anchor_circle = self.create_circle(self.anchor_points[-2],
                #                             self.inlet_dir,
                #                             self.in_circle_d/2)
                stem_circle_list = []
                stem_circle_list = self.populate_seg_w_circles(n_div = 10,
                                                               p1 = self.pipe_stem_list[0],
                                                               p2 = self.anchor_points[-1],
                                                               st_circ = self.in_circle_d/2.0,
                                                               end_circ = self.in_circle_d/2.0,
                                                               circ_list = stem_circle_list)                        
                
                branch_circles = branch_circles + stem_circle_list
                self.loft_circle_surfaces(branch_circles)


    

    ############### HELPERS ###################

    def create_line_seg_list(self,origin,seg_len,direction):
        '''
        Creates a list of 2 points based on origin and a direction + length

        '''
        seg_list = []
        seg_list.append(origin)
        seg_list.append(Point.Create(M(seg_len)*direction[0],
                                     M(seg_len)*direction[1],
                                     M(seg_len)*direction[2]))
        return seg_list
    

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

    # def dividing_curve(self, curve, num_divisions):

    # #curve: A SpaceClaim curve(Curve Object)
    # #num_divisions: How many points on the curve would you like returned.(Float)
    #     divided_curve_points = []
    #     for i in range(num_divisions+1):
    #         i = float(i)
    #         evalu = curve.Evaluate(1.0-(i/num_divisions))
    #         divided_curve_points.append(evalu.Point)
    #     return divided_curve_points #Returns a list of point objects on the curve.

    #Input a curve and a number between 0-1. This equation will give you a point on that curve and the tangent direction of that point.
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

    def populate_seg_w_circles(self,n_div,p1,p2,st_circ,end_circ,circ_list):
        '''
        - populate curve with circles that interpolate from start radius to 
          end radius along the curve

        '''
        for i in range(n_div):
            eval_pt = (1.0-float(i+1)/float(n_div))
            pt_loc = Point.Create(p1.X + eval_pt*(p2.X - p1.X),
                              p1.Y + eval_pt*(p2.Y - p1.Y),
                              p1.Z + eval_pt*(p2.Z - p1.Z))
            direction = Direction.Create(p2.X - p1.X,
                                         p2.Y - p1.Y,
                                         p2.Z - p1.Z)
            pt_radii = self.radius_interpolate(st_circ, end_circ, eval_pt)
            circ_list.append(self.create_circle(pt_loc, direction, pt_radii))
        
        return circ_list

    # def clean_face(self,location,direction,offset,D,depth):
    #     '''


    #     '''
    #     inlet_pt = Point.Create(M(location.X-offset),
    #                             M(location.Y),
    #                             M(location.Z))

    #     circle = self.create_circle(inlet_pt,
    #                                 direction,
    #                                 D/2.0 + MM(10.0))

    #     selection = FaceSelection.Create(circle.Faces[0])
    #     options = ExtrudeFaceOptions()
    #     options.ExtrudeType = ExtrudeType.Cut
    #     result = ExtrudeFaces.Execute(selection, offset + depth, options)


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
            out_faces = self.name_faces('outlet',self.outlet_list)
            body_faces = self.name_body_faces(in_faces,out_faces)
        
        elif self.face_mode == 'reverse':
            in_faces = self.name_faces('inlet',self.outlet_list)
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
        split_pipe_{face_mode}_A_in_{inlet_area}_A_out_{outlet_area}.scdoc
        
        '''
        save_file_str = 'split_pipe' + '_' + \
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

    # def translate_body(self,direction):
    #     '''
    #     translate the body in a given direction

    #     '''
    #     for i in range(len(GetRootPart().GetBodies())):
    #         if GetRootPart().Bodies[i].GetName() == 'Solid':
    #             bod = GetRootPart().Bodies[i]

    #     selection = BodySelection.Create(bod)
    #     options = MoveOptions()
    #     result = Move.Translate(selection, direction, options)





########################## Routine ##########################



pipe_dict = {

'pipe_d_var' : .190,
'pipe_d_min' : .010,

# is a multiple of 1 to 3 lengths of the inlet pipe diameter
'pipe_stem_var' : 3,
'pipe_stem_min' : 1,

# plane that the outlet points will reside on
'outlet_plane_var'   : 3,
'outlet_plane_min'   : 2,

'pipe_mode': 'interpolate',
'face_mode': 'normal',

'save_dir': save_dir
}


i = 0
while i < n_geo_to_build:
        
    pipe = SplitPipe(pipe_dict)
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



import clr
clr.AddReference("Microsoft.Office.Interop.Excel")
import Microsoft.Office.Interop.Excel as Excel


def read_flow_params(excel_file):

    ''' 
    reads the flow params from the csv

    '''
    # Open Excel and the workbook
    ex = Excel.ApplicationClass()
    ex.Visible = False
    workbook = ex.Workbooks.Open(excel_file) 
    worksheet=workbook.ActiveSheet

    # Pull out the size of our data in the excel sheet
    num_col = worksheet.UsedRange.Columns.Count
    num_row = worksheet.UsedRange.Rows.Count

    # grab the data from the worksheet
    output_list = [[],[],[],[],[]]
    for i in range(1,num_row+1):
        for j in range(1,num_col+1):
            output_list[j-1].extend([float(worksheet.Cells(i,j).Value2.ToString())])

    # close the worksheet
    workbook.Close(excel_file)
    ex.Quit()
    return output_list


def read_D(fpath):
    '''
    - Read inlet and outlet diameters from the D_file.txt file
    - Convert to floats and return
    '''

    in_list = []
    out_list = []
    f = open(fpath)
    f_string = f.read()
    f.close()
    f_string = f_string.splitlines()
    in_values = f_string[0].split(':')[1].split(',')
    out_values = f_string[1].split(':')[1].split(',')

    for entry in in_values:
        if len(entry) > 0:
            in_list.extend([float(entry)])

    for entry in out_values:
        if len(entry) > 0:
            out_list.extend([float(entry)])

    return in_list,out_list


def create_log_file(fpath,title_str):
    with open(fpath,'w') as f:
        f.write(title_str)

    return

def update_log_file(fpath,update_str):
    with open(fpath,'a+') as f:
        f.write(update_str)
    
    return

def create_simulation_status_file(fpath):
    with open(fpath,'w') as f:
        f.write("1")

def set_simulation_status_fail(fpath):
    with open(fpath,'w') as f:
        f.write("0")



def flow_params_str(**params):
    
    out_str = ''
    out_str += 'Re: ' + str(params['Re']) + ','
    out_str += 'Eps: ' + str(params['Eps']) + ','
    out_str += 'Visc: ' + str(params['Visc']) + ','
    out_str += 'Rho: ' + str(params['Rho']) + ','
    out_str += 'P: ' + str(params['P']) + ','
    out_str += 'V: ' + str(params['V']) + ','
    out_str += 'D: ' + str(params['D']) + ','
    out_str += 'A_in: ' + str(params['A_in']) + ','
    out_str += 'A_Out: ' + str(params['A_out']) + ','
    out_str += 'T_int: ' + str(params['T_int'])
    out_str += '\n'
    return out_str

def msh_cnt_to_str(fpath,n_nodes,n_elem):
    msh_count = 'N_Nodes: ' + str(n_nodes) + ',' + 'N_Elem: ' + str(n_elem) + '\n'
    return msh_count


def mesh_qual_metrics_str(Min,Max,Avg,Std):
    
    qual_str= ''
    qual_str+= 'Min: ' + str(Min) + ','
    qual_str+= 'Max: ' + str(Max) + ',' 
    qual_str+= 'Avg: ' + str(Avg) + ',' 
    qual_str+= 'Std: ' + str(Std) 
    qual_str+= '\n'

    return qual_str


def mesh_var_to_text(D,BL,log_path):
    '''
    provides all of the variables to the core meshing script
    
    '''

    output_str = ''
    output_str = output_str + 'D = ' + str(D)  + '\n'
    output_str = output_str + 'BL = ' + str(BL) + '\n'
    output_str = output_str + 'log_path = ' + str(log_path) + '\n'
    return output_str

def gen_var_to_text(var,var_str):

    output_str = ''
    output_str += output_str + var_str + ' = ' + str(var) + '\n'
    return output_str

def gen_path_var_to_text(fpath,var_str):
    '''
    takes a path string and returns the string with forward slashes so it can 
    be passed as part of a larger string

    i.e var_str = 'File_path'
        fpath   = 'C:\\path_to_file'
        function returns 'File_path = \'C:\\path_to_file\' \n'

    '''

    output_str = ''
    output_str += output_str + var_str + ' = ' + '\'' + fpath + '\'' + '\n'
    return output_str


def calc_turb_int(Re):
    '''
    equation to calculate turbulent inlet intensity:
    - derived from Ansys fluent theory guide 
    '''
    return 0.16*((Re)**(-1.0/8.0))*100.0 # return value in percent

def grab_inlet_outlet_area(file_string):
    '''
    
    pull inlet and outlet area from the geometry file_string

    '''
    areas = file_string.split('_A_in_')[1]
    A_in, A_out = areas.split('_A_out_')[0], areas.split('_A_out_')[1]
    
    A_in = float(A_in)
    
    A_out = A_out.split('.scdoc')[0]
    A_out = float(A_out)

    return A_in, A_out














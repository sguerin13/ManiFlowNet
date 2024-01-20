import os
import data.wrangling.data_utils as DU
from scripts.helpers import load_config

if __name__ == '__main__':
	config = load_config(os.path.join("scripts","data","scaler_spot_check.json"))

	ndim_scaler_path = config.ndimScalerPath
	norm_scaler_path = config.normScalerPath

	#--------------- Non-Dim Eps and Rho -----------------------#
	velo_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'velo_scaler.pkl'))
	pressure_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'P_scaler.pkl'))
	fp_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'fluid_prop_scaler.pkl'))
	bbox_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'bbox_scaler.pkl'))
	values_included_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'props_included.pkl'))

	print("NON_DIMENSIONALIZED SCALERS - Re and Eps As Context")
	print(" Values Included: ", values_included_scaler)
	print(" Velocity Scaler: ")
	print("\tData Range: ",velo_scaler.data_range_,", N_feats In: ", velo_scaler.n_features_in_,", Scale: ", velo_scaler.scale_)
	print(" Pressure Scaler: ")
	print("\tData Range: ", pressure_scaler.data_range_,", N_feats In: ", pressure_scaler.n_features_in_,", Scale: ", pressure_scaler.scale_)
	print(" Fluid Props Scaler: ")
	print("\tData Range: ",fp_scaler.data_range_,", N_feats In: ", fp_scaler.n_features_in_,", Scale: ", fp_scaler.scale_)
	print(" Bounding Box Scaler: ")
	print("\tData Range: ",bbox_scaler.data_range_,", N_feats In: ", bbox_scaler.n_features_in_,", Scale: ", bbox_scaler.scale_)

	'''
	Output as of 4/29/23:
	NON_DIMENSIONALIZED SCALERS - Re and Eps As Context
	Values Included:  ['Re', 'Eps']
	Velocity Scaler: 
		Data Range:  [182.02408068 182.02408068 182.02408068] , N_feats In:  3 , Scale:  [0.01098756 0.01098756 0.01098756]
	Pressure Scaler: 
		Data Range:  [4327.97895098] , N_feats In:  1 , Scale:  [0.00023105]
	Fluid Props Scaler: 
		Data Range:  [1.56626737e+05 9.95535878e+00] , N_feats In:  2 , Scale:  [6.38460597e-06 1.00448414e-01]
	Bounding Box Scaler: 
		Data Range:  [4.581555] , N_feats In:  1 , Scale:  [0.21876535]

	'''


	#--------------- Non-Dim All Params -----------------------#

	velo_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'velo_scaler.pkl'))
	pressure_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'P_scaler.pkl'))
	fp_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'fluid_prop_scaler.pkl'))
	bbox_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'bbox_scaler.pkl'))
	values_included_scaler = DU.load_scaler(os.path.join(ndim_scaler_path,'props_included.pkl'))
	print("\n\n\n\nNON_DIMENSIONALIZED SCALERS - All Fluid Params As Context")
	print(" Values Included: ", values_included_scaler)
	print(" Velocity Scaler: ")
	print("\tData Range: ",velo_scaler.data_range_,", N_feats In: ", velo_scaler.n_features_in_,", Scale: ", velo_scaler.scale_)
	print(" Pressure Scaler: ")
	print("\tData Range: ", pressure_scaler.data_range_,", N_feats In: ", pressure_scaler.n_features_in_,", Scale: ", pressure_scaler.scale_)
	print(" Fluid Props Scaler: ")
	print("\tData Range: ",fp_scaler.data_range_,", N_feats In: ", fp_scaler.n_features_in_,", Scale: ", fp_scaler.scale_)
	print(" Bounding Box Scaler: ")
	print("\tData Range: ",bbox_scaler.data_range_,", N_feats In: ", bbox_scaler.n_features_in_,", Scale: ", bbox_scaler.scale_)

	'''

	NON_DIMENSIONALIZED SCALERS - All Fluid Params As Context
	Values Included:  ['Re', 'Eps', 'Visc', 'Rho', 'D']
	Velocity Scaler: 
		Data Range:  [182.02408068 182.02408068 182.02408068] , N_feats In:  3 , Scale:  [0.01098756 0.01098756 0.01098756]
	Pressure Scaler: 
		Data Range:  [4327.97895098] , N_feats In:  1 , Scale:  [0.00023105]
	Fluid Props Scaler: 
		Data Range:  [1.56626737e+05 9.95535878e+00 1.39970000e+00 2.52000000e+03
	1.89270038e-01] , N_feats In:  5 , Scale:  [6.38460597e-06 1.00448414e-01 7.14438808e-01 3.96825397e-04
	5.28345644e+00]
	Bounding Box Scaler: 
		Data Range:  [4.581555] , N_feats In:  1 , Scale:  [0.21876535]

	'''


	#--------------- Physical Quantities All Params --------------#
	velo_scaler = DU.load_scaler(os.path.join(norm_scaler_path,'velo_scaler.pkl'))
	pressure_scaler = DU.load_scaler(os.path.join(norm_scaler_path,'P_scaler.pkl'))
	fp_scaler = DU.load_scaler(os.path.join(norm_scaler_path,'fluid_prop_scaler.pkl'))
	bbox_scaler = DU.load_scaler(os.path.join(norm_scaler_path,'bbox_scaler.pkl'))
	values_included_scaler = DU.load_scaler(os.path.join(norm_scaler_path,'props_included.pkl'))
	print("\n\n\n\nPhysical Quantities - All Params As Context")
	print(" Values Included: ", values_included_scaler)
	print(" Velocity Scaler: ")
	print("\tData Range: ",velo_scaler.data_range_,", N_feats In: ", velo_scaler.n_features_in_,", Scale: ", velo_scaler.scale_)
	print(" Pressure Scaler: ")
	print("\tData Range: ", pressure_scaler.data_range_,", N_feats In: ", pressure_scaler.n_features_in_,", Scale: ", pressure_scaler.scale_)
	print(" Fluid Props Scaler: ")
	print("\tData Range: ",fp_scaler.data_range_,", N_feats In: ", fp_scaler.n_features_in_,", Scale: ", fp_scaler.scale_)
	print(" Bounding Box Scaler: ")
	print("\tData Range: ",bbox_scaler.data_range_,", N_feats In: ", bbox_scaler.n_features_in_,", Scale: ", bbox_scaler.scale_)


	'''

	Physical Quantities - All Params As Context
	Values Included:  ['Re', 'Eps', 'Visc', 'Rho', 'D']
	Velocity Scaler: 
		Data Range:  [263.35724 263.35724 263.35724] , N_feats In:  3 , Scale:  [0.00759425 0.00759425 0.00759425]
	Pressure Scaler: 
		Data Range:  [22797186.] , N_feats In:  1 , Scale:  [4.3865063e-08]
	Fluid Props Scaler: 
		Data Range:  [1.56626737e+05 9.95535878e+00 1.39970000e+00 2.52000000e+03
	1.89270038e-01] , N_feats In:  5 , Scale:  [6.38460597e-06 1.00448414e-01 7.14438808e-01 3.96825397e-04
	5.28345644e+00]
	Bounding Box Scaler: 
		Data Range:  [4.581555] , N_feats In:  1 , Scale:  [0.21876535]

	'''
# RF_Precipitation_Grid
Random Forest grid generation:

1) Read the netcdf of the data and extract values against dates. 
These values, for each model, are saved in “name_model.mat” (with the original resolution), 
being name_model, the corresponding entry (MSWEP, CDT, etc.). See script 01_script.py.

2) The “name_model.mat” data obtained in the previous step is interpolated, 
using the Nearest Neighbor method, at 5 km resolution, leaving the new 
“name_model_interp_nearest.mat” files. See script 03_script.py.

3) Once the input models (X1, X2, …, Xn) have been interpolated, the eval.py script is executed, 
in which the possible combinations for RF training are generated. 
Once trained, each RF model is saved in the “models/” directory, with the name “model_X1_X2_Y_rf_trained.pkl”, 
being X1,X2,Y the two predictor models and the paradigm, respectively. Additionally, the metrics obtained during 
the training are saved in text format, as well as the annual march, and optionally, the visual output of the resulting maps. 
The results of each combination are saved in netCDF format (“X1_X2_Y_RF.nc”).

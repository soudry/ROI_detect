Files in folder:

*Main Functions:
RunFista - run the FISTA algorithm for solving group lasso
RunSVDmethod - run coordinate descent gready pca to find neuronal activity and (non-negtive) shapes 
SimulateData

* Auxilarry functions for RunFista:
GetActivity
GetBox
GetCenters
GetROI
MSEBounds - get upper and lower bounds for MSE on datasets
imblur - Apply Gaussian blur on an array

* Auxilarry functions for NMFmethod:
RegionAdd
RegionCut
RegionInsert

*Misc Functions:
GetFileName - generate a file name for datasets and parameters
GetHomeFolder - generate a home folder from which we load data and save results in
v2struct - transform struct into variables

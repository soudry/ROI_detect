
 - Most of the code is currently in Matlab:
Files: 
* Demo.m - Run with option 1, and then option 2 to see how code works
* RunFISTA.m -  runs the "group lasso" code to detect neuronal centers and activity
* RunNMFmethod - run coordinate descent gready NMF to find neuronal activity and (non-negtive) shapes, based on group lasso initialization
* GetDefaultInput -  fetch default parameters for each dataset (params,flags,specs)

Folders:
* Misc - Auxilary Functions
* Plotting - Auxilary Functions for plotting only
* Datasets - some datasets for demonstrations

 - Part of the code had been has been converted to matlab
 * GetNeuronCenters.py - a pythonized version of RunFISTA.m. The code has been simplified, and some modifications have been made.
 For example, when adapting the regularization cosntant, the binary search was replaced with an exponential search. 
 Also, the background substraction has been removed, as it probably can be made more efficiently on thunder.



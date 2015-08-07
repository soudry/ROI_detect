The code is available in Matlab, Python and Thunder(Spark)

-- Matlab --
Files: 
* Demo.m - Run with option 1, and then option 2 to see how code works
* RunFISTA.m -  runs the "group lasso" code to detect neuronal centers and activity
* RunNMFmethod.m - run coordinate descent gready NMF to find neuronal activity and (non-negtive) shapes, based on group lasso initialization
* GetDefaultInput.m -  fetch default parameters for each dataset (params,flags,specs)

Folders:
* Misc - Auxilary Functions
* Plotting - Auxilary Functions for plotting only
* Datasets - some datasets for demonstrations


-- Python --
Files: 
* Demo.py - Run with option 1, 2 or 3 to see how code works
* BlockGroupLasso.py  - a pythonized version of RunFISTA.m
* BlockLocalNMF.py - a pythonized version of RunNMFmethod.m


-- Thunder (https://github.com/j-friedrich/thunder/tree/LocalNMF) --
File:
* DemoThunder.ipynb - Change option to 1, 2 or 3 in ipython notebook to see how code works
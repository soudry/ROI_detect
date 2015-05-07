% 16/4/2014
clear all;
close all
clc

addpath('Misc')
addpath('DataSets')
addpath('Plotting')
%% Set inputs - choose dataset
option=2;

if option==1
    data_name='Simulated';  %simulated 2D data
    load('data_sim.mat') %load data - an XxYxT   movie
elseif option==2
    data_name='Exp2D';  % Experimental 2D data
    load('data_exp2D.mat') %load data - an XxYxT movie
elseif option==3
    data_name='Exp3D'; % Experimental 3D data  (only group lasso part is relevant here, due to low temporal resolution)
    load('data_exp3D.mat') %load data - an XxYxZxT movie
end

data=double(data);
[specs,params,flags]=GetDefaultInput(data_name); % get algorithm parameters
%% Run Fista
[x, u,f, neuron_stats,performance, FISTA_params ] = RunFISTA(data,params,flags,specs); 

%% Plot FISTA
if option==3
    PlotPerformance(performance) ;
    PlotCenters(data,neuron_stats.cent);
    show_contours=0;
    VideoCenters(data,neuron_stats,data_name,show_contours)
else
    %% Run Non-Negative Matrix factorization part
    bias=bsxfun(@times,u,f);   
    [MSE_array,v_s,v_t,box_cell]=RunNMFmethod(data-bias,params,flags,neuron_stats);
     
    %% Plot NMF 
    plot(MSE_array);   ylabel('MSE');   xlabel('iterations');
    PlotAllNeurons(v_s,v_t,box_cell,[]);
    PlotEachNeuron(data,v_s,v_t,box_cell)
end
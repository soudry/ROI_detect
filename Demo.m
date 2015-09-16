% 16/4/2014
clear all;
close all
clc

addpath('Misc')
addpath('DataSets')
addpath('Plotting')
%% Set inputs - choose dataset
option=5;

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

[specs,params,flags]=GetDefaultInput(data_name); % get algorithm parameters
%% Run Fista
[x, u,f, neuron_stats,performance, FISTA_params ] = RunFISTA(data,params,flags,specs); 

%% Plot FISTA
%     PlotPerformance(performance) ;
%     PlotCenters(data,neuron_stats.cent);
%     show_contours=0;
%     VideoCenters(data,neuron_stats,[],show_contours)

%% Run Non-Negative Matrix factorization part
[MSE_array,v_s,v_t,box_cell,neurons_removed]=RunNMFmethod(data,params,flags,neuron_stats,u,f);
%% Find new "center of mass" centers according to NMF
L=(length(v_s)-1);
cent_new=zeros(L,ndims(data)-1);
for ll=1:(length(v_s)-1)
        box=box_cell{ll};
        size_v=box(:,2)-box(:,1)+1;
        cent_new(ll,:)=round(centerOfMass(reshape(v_s{ll},size_v'))+box(:,1)-1)';
end

    %% Plot NMF 
    figure(1002); plot(MSE_array);   ylabel('MSE');   xlabel('iterations');

    PlotCenters(data,cent_new);
    PlotAllNeurons(v_s,v_t,box_cell,data_name)
%     PlotEachNeuron(data,v_s,v_t,box_cell)
    [estimate,bias]=GetDenoisedData(size(data),v_s,v_t,box_cell,flags.adapt_bias);
%     VideoResidual(data,estimate,[],bias)
    VideoCentersandResidual(data,v_s,v_t,box_cell,neuron_stats.cent',1:6,estimate,data_name,bias)

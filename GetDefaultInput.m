function [specs,params, flags]=GetDefaultInput(data_name)
%% input
% data_name - name of dataset (this is used as a switch to decide which paramters to use)
%% ouput (see details in code below)
% specs- dataset specifications (struct)
% params - algorithms parameters (struct)
% flags - algorithms control flags (struct)
%% Some guidlines on parameter adjustment: 
% * sigma - size of neuron (formally, width of Gaussian used to model neuron shape) 
% * sigma_vector  - size of neuron on each axis (useful when axes pixel resolutions are different). 
% make sure size of sigma_vector is also the spatial size of the data
% * lambda - increase to make result sparser (less neurons)
% * adapt_bias - if 1, the algorithm remove background fluctuations (rank-1 approximation) from the video
% * adapt_lambda - if hard to find lambda, set to 1. This should adapt lambda (starting
% from initial condition lambda) to satisfy the targets in one of two next parameters. 
% Note though it could take a lot of iterations, so sometime it better to change lambda manually in the right direction (the direction of the adaptation)
% * TargetAreaRatio - perscentage of picture covered by neurons. if equal [], use next parameter as target
% * Neuron_number_range - possible range of neuron numbers
%% default values

%control flags
adapt_bias=0; % use FISTA with non-stationary basline removal
adapt_lambda=0; % do SGD on lambda so we get target sparisity level or neuron numbers
non_negative=0; %inforce non-negativity on x?
save_results=1; % save_results?
save_each=0; % flag to save: 'each' iteration
save_x=0; %should we save x in FISTA?

%params        
TargetAreaRatio=[]; % percentage of spatial area that infered weights cover, if empty algorithm uses Neuron_number_range
iterations=10; % number of Fista iterations
prev_iteration=0; % if not zero, load previous run with same sigma and lambda. 
lambda=0.002;  % regularization constant (only initial guess if adapt_lambda=1)
eta=0.5; %lambda learning rate when we adapt lambda- is there a better way to set it?
sigma=6; %hsize=sigma*10;       
sigma_vector=sigma*[1 1]; %only useful for 3D data
Neuron_number_range=[4 4]; %possible range of neuron numbers

%obselete - don't use (I keep these for software development purposes)
centralize=1;  %should we centralize remove the mean from the data?
normalize=1;   %should we normalize the data to be in [0,1]
show_ground_truth=0; % use ground truth data?
cut_box=[];     %should we cut parts from the data?
K=[];           %if not empty, average data over time with window of size K
g=0.76; %decay constant for correlations in data. if filter_time=1 we filter with this
lambda_sparse=1; %% regularization constant for sparsity
filter_time=0; % should we filter x in time to get activity?
make_sparse=filter_time; %should x be temporally sparse?

switch data_name
        
    case 'Simulated'
        %use default values
    case 'Exp2D'                     
        TargetAreaRatio=0.01; % percentage of spatial area that infered weights cover
        lambda=1;  % regularization constant
        eta=0.5; %lambda learning rate when we adapt lambda- is there a better way to set it?
        g=0.76; %decay constant for correlations in data. if filter_time=1 we filter with this
        sigma=5; %hsize=sigma*10;       
        sigma_vector=sigma*[1 1]; %only useful for 3D data
    case 'Exp3D'        
        non_negative=1; 
        save_results=1; % save_results?
        TargetAreaRatio=0.01; % percentage of spatial area that infered weights cover, if empty algorithm uses Neuron_number_range
        adapt_lambda=1; % do SGD on lambda so MSE-> MSE_bottom
        sigma=2; %hsize=sigma*10;       
        sigma_vector=sigma*[1 1 1]; %only useful for 3D data
    otherwise
        error('unknwon dataset');
end

with_bias=1-adapt_bias; % add squared mean to MSE_bottom

specs=v2struct(data_name,normalize,centralize,cut_box,K,show_ground_truth);
flags=v2struct(adapt_bias,adapt_lambda,with_bias,save_each,filter_time,make_sparse,non_negative,save_x,save_results);
params=v2struct(data_name,sigma_vector,g,lambda,lambda_sparse,prev_iteration,iterations,eta,TargetAreaRatio,Neuron_number_range);


        
end
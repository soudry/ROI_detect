%% 16/4/2014

function [x, u, f, neuron_stats, performance, FISTA_params ] = RunFISTA(data,params,flags,specs) %#ok
%% inputs:
% data
% 2D or 3D video of calcium imaging in neurons (XxYxT or XxYxZxT array)

%params - algorithm parameters
v2struct(params)
% data_name : name of dataset
% iterations : number of Fista iterations
% prev_iteration:  if not zero, load previous run with same sigma and lambda. 
% lambda  : regularization constant for group lasso
% eta : lambda learning rate when we adapt lambda 
% sigma_vector : standard deviation of Gaussian blurs
% TargetAreaRatio : required area covered by infered x 

% flags - control flags
v2struct(flags)
% adapt_bias : use FISTA with bias?
% adapt_lambda : do SGD on lambda 
% with_bias : add squared mean to MSE_bottom ?
% non_negative: inforce non-negativity on x?
% save_results: save results?
% save_x: %should we save x in FISTA?
% save_each; : flag to save 'each' iteration ?

% specs
% data specs - note important (code just saves this info with data)


%% Outputs
% x - The infered neuronal activity  (XxYxT or XxYxZxT array)
% u - Baseline  (XxY or XxYxZ  array)
% neuron_stats - a struct containing L infered "neurons": 
%       cent - Lx2 or Lx3 array with neuronal centers
%       ROI - an XxY or XxYxZ array with Regions Of Intrest correponding to each neuron (mostly zeros, with the regions value is the  index of the corresponding neuron)
%       activity - LxT array with activity trace for each ROI
% performance - a struct containing information on the performance of the FISTA algorithm:
%       MSE_array - mse error (residual): ||x-data||^2, at each itetation of FISTA
%       MSE_top and MSE_bottom - naive bounds on the MSE error
%       sparsity_ratio - fraction pixels in x which are non zero (at least for some t), at each itetation of FISTA
%       Error_array - the FISTA objective, at each itetation of FISTA
%       t_elapsed - simulation time in sec
% FISTA_params - a few algorithm parameters from params input. Note lambda and lambda_sparse may change after running the algoirhtm


%%%%%%%%%%%%%%
% set up saving folders
homefolder = GetHomeFolder;
folder=fullfile( homefolder, 'FISTA_Results',data_name);
if save_results
    if ~exist(folder,'dir') % make folder for dataset if does not exist
        mkdir(folder)
    end
end
file_name = GetFileName(params);
fullpath=fullfile(folder,file_name);

%  initialize
dims=size(data);
T=dims(end); % Time
D=length(dims); %number of dimensions
if size(sigma_vector)~=D-1
    error('size of params.sigma_vector does not match spatial dimensions of data')
end

% Calc MSE bounds - an estimate that helps set lambda
[MSE_top, MSE_bottom] = MSEbounds( data, with_bias);

% initialization for optimization
if prev_iteration==0 
    x=data*0;
    if adapt_bias  %B=(1,1,...,1)/T
        p=0.3; %initial guess for background is the 30% quantile
        u=quantile(data,p,D); 
        [u,f] = GreedyNNPCA(data,u);
    else
        u=0; f=0;        
    end
    t_initial=0; %time of simulation

    %intialize arrays
    MSE_array=zeros(iterations,1);
    Error_array=zeros(iterations,1); %MSE and L1 error together
    sparsity_ratio=MSE_array;    
else    
    name_prev=[fullpath '_k=' num2str(prev_iteration) '.mat'];
    load(name_prev,'x','u','f','params','flags','performance','neuron_stats','t_elapsed','iterations_so_far');
    t_initial=t_elapsed;
    v2struct(performance) 
    lambda=params.lambda;
    lambda_sparse=params.lambda_sparse;
end
bias=bsxfun(@times,u,f);

if ~strcmp(computer,'GLNXA64') %if not on Yeti...
    if gpuDeviceCount>1; %if this computer have cuda support, use it
        data=gpuArray(data);
        x=gpuArray(x);
        u=gpuArray(u);
    end
end

t_next=1;
y=x;

% precompute some stuff
L=2;
win_size=max(sigma_vector)*5; %Gaussian window width
gauss_conv=@(z) imblur(z,sigma_vector,win_size,D-1);

%main loop - FISTA
tic
for kk=1:iterations
    iterations_so_far=prev_iteration+kk;
    %[x u y z t] = FISTA_Blur(sigma,data,x,y,u,z,t,lambda,T); %with bias
    %initialize             
        t=t_next;
        x_prev=x;

        r=gauss_conv(y);
        q=y-(2/L)*gauss_conv(r+bias-data);
         
        % Shrinkage
        if adapt_bias
            update_time=5; %update bias every update_time
            if mode(kk,update_time)+1==update_time
                [u,f] = GreedyNNPCA(data-r,u);
                bias=bsxfun(@times,u,f);            
            end
        end
                    
        if non_negative
            q(q<0)=0;
        end
        
        if make_sparse %sprasify using threshold operator
            temp=(1-(lambda_sparse/L)./abs(q));   
            temp(temp<=0)=0;
            q=temp.*q;
        end
    % ThresholdOperator(u,lambda/L,groups);
        temp=(1-(lambda/L)./sqrt(sum(q.^2,D)));   
        temp(temp<=0)=0;
        x=bsxfun(@times,temp,q);
        
    % Some more Fista stuff
        t_next=(1+sqrt(1+4*t^2))/2;
        y=x+((t-1)/t_next)*(x-x_prev);

    % Adaptive restart from Donoghue2012    
    do_restart=(mean2((q-y).*(x-x_prev)))>0);
    if do_restart
        t_next=1;
        y=x;        
    end
        
    t_elapsed=t_initial+toc;
    
    %% update performance indicatiors
    temp=bias+r;
    MSE_total=mean2((temp-data).^2);   
    MSE_array(iterations_so_far)=MSE_total;
    
    temp=sum(sqrt(sum(x.^2,D)));
    Error_array(iterations_so_far)=MSE_array(iterations_so_far)+lambda*sum(temp(:))/prod(dims);
    if make_sparse
        Error_array(iterations_so_far)=Error_array(iterations_so_far)+lambda_sparse*sum(abs(x(:)))/prod(dims);
    end
    % check sparsity_ratio
    temp=sum(abs(x),length(dims));
    sparsity_ratio(iterations_so_far)=sum(temp(:)>0)/prod(dims(1:(end-1)));    
    
    % find number of neurons
    pic=std(x,[],length(dims));           
    cent=GetCenters(pic); %find neuronal centers
    num_neurons=size(cent,1) %#ok
    
    % adapt lambda    

        if adapt_lambda
            if ~isempty(params.TargetAreaRatio) %just use sparsity ratio    
                    AreaRatio=sparsity_ratio(iterations_so_far)
                    lambda=lambda*(1-params.TargetAreaRatio*tanh((params.TargetAreaRatio-AreaRatio)/params.TargetAreaRatio))
            else %old options
%                 
                if kk>1
                    prev_prev_lambda=prev_lambda;
                end
                prev_lambda=lambda;                    

            if sparsity_ratio(iterations_so_far)>0.9
                lambda=lambda/(1-eta);
            elseif Neuron_number_range(1)>num_neurons %too few neurons
                lambda=lambda*(1-eta) %#ok               

            elseif Neuron_number_range(2)<num_neurons %too many neurons
                if sparsity_ratio(iterations_so_far)==0
                    lambda=lambda*(1-eta)
                else
                    lambda=lambda/(1-eta) %#ok
                end 
            else %find MSE    
                
                relative_error=(MSE_array(iterations_so_far)-MSE_bottom)/MSE_bottom %#ok
                lambda=lambda*(1-0.1*eta*tanh(relative_error)) %#ok
            end
                if kk>2
                    if (lambda-prev_lambda)*(prev_lambda*prev_prev_lambda)<0 %if going around fixed point, decrease learning rate (but only if these fluctuations are related to lambda...)
                        if sparsity_ratio(iterations_so_far)<1
                            if sparsity_ratio(iterations_so_far)>0
                                eta=eta*0.9;
                            end
                        end
                    end
                end
            end
        end

        disp([ 'Fista iteration #' num2str(iterations_so_far)])
    %
    
    
    %% save    
    name_current=[fullpath '_k=' num2str(iterations_so_far) '.mat'];
    name_prev=[fullpath '_k=' num2str(iterations_so_far-1) '.mat'];
    params=v2struct(sigma_vector,g,lambda,lambda_sparse,iterations,eta,TargetAreaRatio);   
    performance=v2struct(Error_array,MSE_array,MSE_top,MSE_bottom,sparsity_ratio,t_elapsed); 
    
    if save_each||(kk==iterations)||((~adapt_lambda)&&all(~x(:)))
        ROI=[]; activity=[];
        Max_neurons_num=300;
        if num_neurons<Max_neurons_num
            cent=GetCenters(pic); %find neuronal centers
            ROI = GetROI(pic,cent); %find neuronal ROI
            activity = GetActivity(x,ROI); %find neuronal activity                
        else
            warning(['Neurons number is larger than Max_neurons_num=' num2str(Max_neurons_num) '. ROI and activity not saved.'])
        end
        neuron_stats=v2struct(cent,ROI,activity); %#ok %neuronal statistics 
        
        if save_results
            if save_x
                save(name_current,'x','u','f','params','flags','specs','performance','neuron_stats','t_elapsed','iterations_so_far');
            else
                save(name_current,'u','f','params','flags','specs','performance','neuron_stats','t_elapsed','iterations_so_far');
            end
        end
    end
    if save_results&&~save_each&&(kk>1)%do not delete starting point
        delete(name_prev);
    end  
    
    
    if (~adapt_lambda)&&all(~x(:))        
        warning('no neurons found')
        return
    end

end

FISTA_params=params;

end
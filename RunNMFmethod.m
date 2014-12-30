function [MSE_array,v_s,v_t,box_cell]=RunNMFmethod(data,params,flags,neuron_stats)
%% Run local Non-negative Matrix Factorization (NMF) method with coordinate descent
%  to extract neuronal shapes using Group lasso centers

% Inputs:
% data - dataset
% params - algorithm params
% flags - algorithm flags
% neuron_stats - a structure with centers, activity and ROI extraced from group lasso

% Outputs:
% MSE_array - MSE of algorithm in each iteration
% v_s - cell array of neuronal shapes vectors
% v_t - cell array of neuronal activity vectors
% box_cell - cell array of specifing the edges boxes in which each neuronal shapes lies

%% Initialize
dims=size(data);  %   dataset dimensions 
R=3*params.sigma_vector; %bounding box for neuron
iterations=params.iterations;
prev_iteration=params.prev_iteration;
data_name=params.data_name;
nn_activity=flags.non_negative; %non-negative activity
file_name= GetFileName(params);
L=size(neuron_stats.cent,1);

v_s=cell(L,1);v_t=cell(L,1);box_cell=cell(L,1);
Z=data;
MSE_array=zeros(iterations,1);

homefolder=GetHomeFolder;
folder=fullfile(homefolder,'NMF_Results', data_name);

name=fullfile(folder,[file_name '_k=']);

for ll=1:L
    box_cell(ll)={GetBox(neuron_stats.cent(ll,:),R,dims)};
    if (prev_iteration>0)
        load([name num2str(prev_iteration) '.mat'],'v_s','v_t','MSE_array');
    else
        %initialize spatial part with a Gaussian Blur centered around center            
        if length(dims)>3;
            [xm, ym, zm]=ndgrid(1:dims(1),1:dims(2),1:dims(3));
            temp=exp( -(((xm-neuron_stats.cent(ll,1)).^2)/(2*params.sigma_vector(1)))...
            -((ym-neuron_stats.cent(ll,2)).^2)/(2*params.sigma_vector(2))...
            -((zm-neuron_stats.cent(ll,3)).^2)/(2*params.sigma_vector(3)) ...
                )/sqrt(2*pi)/prod(params.sigma_vector);
            temp=repmat(temp,[1 1 1 2]); %just so we can use Region Cut
        else
            [xm, ym]=ndgrid(1:dims(1),1:dims(2));
            temp=exp(-((xm-neuron_stats.cent(ll,1)).^2)/(2*params.sigma_vector(1))...
            -((ym-neuron_stats.cent(ll,2)).^2)/(2*params.sigma_vector(2))...
            )/sqrt(2*pi)/prod(params.sigma_vector);
            temp=repmat(temp,[1 1 2]); %just so we can use Region Cut
        end        
        
        temp=RegionCut(temp,box_cell{ll});
        v_s{ll}=temp(:,1);
        %initialize temporal part with activity
        if ~isempty(neuron_stats.activity)
            v_t{ll}=neuron_stats.activity(ll,:)';
        else
            v_t{ll}=ones(dims(end),1);
        end        
    end
        Z=RegionAdd(Z,-v_s{ll}*v_t{ll}',box_cell{ll});
end


%% Main Loop
tic
for kk=1:iterations
    for ll=1:L
        if (prev_iteration>0)||(kk>1)
            Z=RegionAdd(Z,v_s{ll}*v_t{ll}',box_cell{ll});               
        end
        
        %% cut region
        X=RegionCut(Z,box_cell{ll});
        
        %% do SVD        
        greedy_pca_iterations=5;
        for ii=1:greedy_pca_iterations
            temp=X'*v_s{ll}/sum(v_s{ll}.^2);
            if nn_activity
                temp(temp<0)=0;
                v_t{ll}=temp;
            else
                v_t{ll}=temp;
            end
            temp=X*v_t{ll}/sum(v_t{ll}.^2);
            temp(temp<0)=0;
            v_s{ll}=temp;
        end            
        Z=RegionAdd(Z,-v_s{ll}*v_t{ll}',box_cell{ll});
    end
    
    %% Measure MSE
    iteration=prev_iteration+kk;
    disp(['kk=' num2str(kk)]);    
    MSE_array(iteration)=sum(Z(:).^2)/(prod(dims));    
    
    %% save run    
    t_elapsed=toc
    

end
if flags.save_results    
    if ~exist(folder,'dir')
        mkdir(folder)
    end
    save([name num2str(iteration) '.mat'],'MSE_array','v_s','v_t','box_cell','iterations','t_elapsed','params');   
end
end
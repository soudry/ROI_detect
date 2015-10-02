function [MSE_array,v_s,v_t,box_cell,neurons_removed]=RunNMFmethod(data,params,flags,neuron_stats,u,f)
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
% u,f - background spatial and temporal components, if 0, then ignore

%% Initialize
dims=size(data);  %   dataset dimensions 
iterations=params.iterations;
prev_iteration=params.prev_iteration;
data_name=params.data_name;
nn_activity=flags.non_negative; %non-negative activity
file_name= GetFileName(params);
L=size(neuron_stats.cent,1); %background component added

v_s=cell(L,1);v_t=cell(L,1);box_cell=cell(L,1);
Z=data;
MSE_array=zeros(iterations,1);

homefolder=GetHomeFolder;
folder=fullfile(homefolder,'NMF_Results', data_name);

name=fullfile(folder,[file_name '_k=']);

if (prev_iteration>0)
    load([name num2str(prev_iteration) '.mat'],'v_s','v_t','MSE_array');
    L=length(v_s);
else
    M=L;
    if ~and(isscalar(u),u==0) %add background term
        L=L+1;
        box_cell{L}=GetBox(neuron_stats.cent(1,:),3*dims(1:end-1),dims); %whole data sized box
        temp=RegionCut(repmat(u,[ones(1,length(dims)-1) 2]),box_cell{L});
        v_s{L}=temp(:,1);
        v_t{L}=squeeze(f);
        Z=RegionAdd(Z,-v_s{L}*v_t{L}',box_cell{L});
    end
end

for ll=1:M
    % R=round(5*params.sigma_vector); %bounding box for neuron
    R=GetBoxSize(neuron_stats.ROI==ll,params.sigma_vector);
    box_cell(ll)={GetBox(neuron_stats.cent(ll,:),R,dims)};
    if (prev_iteration==0)
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
neurons_removed=[];
for kk=1:iterations
    ll=1;
    while ll<=L
%         if (prev_iteration>0)||(kk>1)
            Z=RegionAdd(Z,v_s{ll}*v_t{ll}',box_cell{ll});               
%         end
        
        %% cut region
        X=RegionCut(Z,box_cell{ll});
        
        %% do SVD        
        greedy_pca_iterations=5;
        flag_removed=0;
        for ii=1:greedy_pca_iterations
            c_s=sum(v_s{ll}.^2);
            centered_cond=1;
            
%             if (ii==greedy_pca_iterations)&&(ll~=L)&&(kk>3)
%                 box=box_cell{ll};
%                 size_v=box(:,2)-box(:,1)+1;
%                 cent_mass=centerOfMass(reshape(v_s{ll},size_v'));
%                 thresh=0.3;
%                 for dd=1:2 %don't check 3D centering                
%                     centered_cond=and(centered_cond,(cent_mass(dd)>size_v(dd)*thresh)&&(cent_mass(dd)<size_v(dd)*(1-thresh))) ;%not too far from the center of the box
%                 end
%             end
            
            if or(c_s==0,~centered_cond)
                v_t(ll)=[];
                v_s(ll)=[];
                box_cell(ll)=[];
                flag_removed=1;
                L=L-1;
                neurons_removed(end+1)=ll+sum(neurons_removed<=ll); %#ok
                break
            end
            temp=X'*v_s{ll}/c_s;
            if nn_activity
                temp(temp<0)=0;
                v_t{ll}=temp;
            else
                v_t{ll}=temp;
            end
            c_t=sum(v_t{ll}.^2);
            if c_t==0
                v_t(ll)=[];
                v_s(ll)=[];
                box_cell(ll)=[];
                flag_removed=1;
                L=L-1;
                neurons_removed(end+1)=ll+sum(neurons_removed<=ll); ; %#ok
                break
            end
            temp=X*v_t{ll}/c_t;
            temp(temp<0)=0;
            v_s{ll}=temp;
        end            
        if flag_removed==0
            Z=RegionAdd(Z,-v_s{ll}*v_t{ll}',box_cell{ll});
            ll=ll+1;
        end
    end
    
    %% Measure MSE
    iteration=prev_iteration+kk;        
    MSE_array(iteration)=sum(Z(:).^2)/(prod(dims));
    figure(3102)
    plot(MSE_array(1:iteration));
    title(['kk=' num2str(kk)]);
    pause(1e-5)
    
    %% save run    
    t_elapsed=toc %#ok
    

end
if flags.save_results    
    if ~exist(folder,'dir')
        mkdir(folder)
    end
    save([name num2str(iteration) '.mat'],'MSE_array','v_s','v_t','box_cell','iterations','t_elapsed','params');   
end
end
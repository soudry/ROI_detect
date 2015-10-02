function R=GetBoxSize(ROI,sigma_vector)
%% Input:
% ROI - binary array in which 1 denotes the detected area for the neurons
% Sigma_vector - size of neurons (group lasso parameter)
%% Output:
% R - box size for localNMF

dims=ndims(ROI);
R=zeros(dims,1);
dims_v=1:dims;
for dd=1:dims
    current_dims_v=dims_v;
    current_dims_v(dd)=[];
    temp=ROI;
    for kk=current_dims_v
        temp=any(temp,kk);
    end
    temp=squeeze(temp);
%     R(dd)=round(sum(temp)*2/3+1.5*sigma_vector(dd));
    R(dd)=round(sum(temp)/2+1.5*sigma_vector(dd));
end
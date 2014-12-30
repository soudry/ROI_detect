function res = RegionCut(X,box,varargin)
%CUTREGION Summary of this function goes here
% inputs:
%  X -  an [XxYxZ...]xT array
%  box - region to cut
%  varargin - specificy dimensions of whole picture (optional)

% outputs:
%  res - Matrix of size prod(R)xT 

%% 
dims=size(X);
if nargin>2
    dims=varargin{1};
end
D=length(dims)-1;

if D~=size(box,1)
    error('box has the wrong number of dimensions');
end

x_ind=box(1,1):box(1,2);
y_ind=box(2,1):box(2,2);

if D>2
    z_ind=box(3,1):box(3,2);
    res=X(x_ind,y_ind,z_ind,:);
    res=reshape(res,[length(x_ind)*length(y_ind)*length(z_ind),dims(end)]);
else
    res=X(x_ind,y_ind,:);
    res=reshape(res,[length(x_ind)*length(y_ind),dims(end)]);
end

%%

end


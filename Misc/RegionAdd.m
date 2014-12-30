function Z= RegionAdd(Z,X,box)
%CUTREGION Summary of this function goes here
% inputs:
%  Z -  dataset [XxYxZ...]xT array
%  box - Dx2 array defining spatial box to put X in
%  X -  Input matrix (prod(diff(box,2))xT)

% outputs:
%  Z=Z+X on box region - a [XxYxZ...]xT array

%% 

dims=size(Z);
D=ndims(Z);
is3D=D>3;

x_ind=box(1,1):box(1,2);
y_ind=box(2,1):box(2,2);

if is3D
    z_ind=box(3,1):box(3,2);
    Z(x_ind,y_ind,z_ind,:)=Z(x_ind,y_ind,z_ind,:)+reshape(X,[length(x_ind),length(y_ind),length(z_ind),dims(end)]);
else
    Z(x_ind,y_ind,:)=Z(x_ind,y_ind,:)+reshape(X,[length(x_ind),length(y_ind),dims(end)]);
end


end


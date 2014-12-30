function ROI = GetROI(pic,cent)
% find ROI and activity for image 'pic' given centers

BW_locations=pic>0;
components=bwlabeln(BW_locations);
dims=size(pic);
is3D=length(dims)>2;
L= size(cent,1);
if is3D
    ROI=zeros(dims(1),dims(2),dims(3));
    distances=zeros(dims(1),dims(2),dims(3),L);
else
    ROI=zeros(dims(1),dims(2));
    distances=zeros(dims(1),dims(2),L);
end

if is3D
    [x_array,y_array,z_array]=ndgrid(1:dims(1),1:dims(2),1:dims(3));
else
    [x_array,y_array]=ndgrid(1:dims(1),1:dims(2));
end
for ll=1:L
    if is3D
        distances(:,:,:,ll)=sqrt((x_array-cent(ll,1)).^2+(y_array-cent(ll,2)).^2+(z_array-cent(ll,3)).^2);
    else        
        distances(:,:,ll)=sqrt((x_array-cent(ll,1)).^2+(y_array-cent(ll,2)).^2);
    end
end

[~, min_dist_ind]=min(distances,[],length(dims)+1);
for ll=1:L
    if is3D
        ind1=components(cent(ll,1),cent(ll,2),cent(ll,3));
        ind2=min_dist_ind(cent(ll,1),cent(ll,2),cent(ll,3));   
    else
        ind1=components(cent(ll,1),cent(ll,2));
        ind2=min_dist_ind(cent(ll,1),cent(ll,2));    
    end

    comp=(components==ind1);
    comp(min_dist_ind~=ind2)=0;
    ROI(comp)=ll;    
end


end


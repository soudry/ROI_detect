function activity = GetActivity(x,ROI)
% Find activity from video x (size (XxYxZ...)xT ) given ROIs (regions of
% intrest) by taking the spatial average in each region

dims=size(x);
L=max(ROI(:));
activity=zeros(L,dims(end));

ROI = reshape(ROI,[prod(dims(1:end-1)) 1]);

x = reshape(x,[prod(dims(1:end-1)) dims(end)]);

Pixels2Del = find(ROI == 0);

ROI(Pixels2Del) = []; x(Pixels2Del,:) = [];

for ll=1:L

 activity(ll,:) = mean(x(ROI==ll,:),1);

end


end


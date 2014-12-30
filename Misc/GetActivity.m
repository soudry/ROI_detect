function activity = GetActivity(x,ROI)
% find ROI and activity for image 'pic' given centers

dims=size(x);
L=max(ROI(:));
activity=zeros(L,dims(end));

for ll=1:L
    for tt=1:dims(end)        
        if length(dims)<4
            temp=x(:,:,tt); 
        else 
            temp=x(:,:,:,tt);              
        end
        activity(ll,tt)=mean(temp(ROI==ll));
    end
end

end


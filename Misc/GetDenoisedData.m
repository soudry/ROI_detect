function [estimate,bias] = GetDenoisedData(dims, v_s,v_t,boxes,adapt_bias)
%GETDENOISEDDATA Summary of this function goes here
% combine together all neurons to generate an estimate of the "denoised
% image" - the combined spatio-temporal activity

%input: 
% dims - data dimensions
% v_s - cell array of shape vectors
% v_t - cell array of trace vectors
% boxes - cell array of shape boxes
% L1 - number of neurons to extract

% output
% estimate - denoise image
% bias - background term

estimate=zeros(dims);
bias=zeros(dims);

L=length(v_s);
if ~adapt_bias
    for ll=1:L
        estimate=RegionAdd(estimate,v_s{ll}*v_t{ll}',boxes{ll});
    end
    disp('No background in the output');
    bias=0*estimate;
else 
    for ll=1:L-1
        estimate=RegionAdd(estimate,v_s{ll}*v_t{ll}',boxes{ll});
    end
%     u=reshape(v_s{L2},dims(1:end-1));    
%     f=reshape(v_t{L2},[ones(1,length(dims)-1),dims(end)]);
        bias=RegionAdd(bias,v_s{L}*v_t{L}',boxes{L});
end



end


function [MSE_top, MSE_bottom] = MSEbounds( data, with_bias)
%FINDMSE Summary of this function goes here
% inputs: 
% data
% with_bias - should we add noise from "mean" activity 
% outputs:
% MSE_bottom - lower bound on MSE
% MSE_top - upper bound on MSE
% find desired MSE (per pixel-time point) from data
%   Detailed explanation goes here

if (with_bias~=0)&&(with_bias~=1)
    error('with_bias should be binary!!')
end

Y=data;
dims=size(Y);
D_s=length(dims)-1; %spatial dimensions
mu=mean(Y,D_s+1);
Y_c=(Y-repmat(mu,[ones(1,D_s) dims(end)]))*((1-1/dims(end))^(-1)); %centralized Y   and renormalize
MSE_top=mean2(Y_c.^2); 


%% Assuming spatial stationarity
for pp = 1:D_s %n-dimensional fft
    Y_c = fft(Y_c,[],pp);
end
psd=abs(Y_c).^2;
norm=prod(dims(1:D_s));
temp=mean(abs(psd),D_s+1);
MSE_H=median(temp(:))/norm; %high frequencies - image baseline variability (the 1/(1-1/T)^2 correction is to offset noise due to substracting basline)

%% Assuming temporal stationarity -too noisy to work
% Y_c = fft(Y_c,[],D_s+1);
% psd=abs(Y_c).^2;
% Noise_var_Y=median(psd,D_s+1)/dims(end);
% MSE_H=mean(Noise_var_Y(:)); %high frequencies - image baseline variability (the 1/(1-1/T)^2 correction is to offset noise due to substracting basline)
% 
% plot(squeeze(mean(mean(psd,1),2)))
%% if with_bias=1
MSE_L=mean2(data)^2; %low frequencies - image baseline intensity
MSE_bottom=with_bias*MSE_L+MSE_H;


end


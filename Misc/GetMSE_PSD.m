function MSE = GetMSE_PSD( Y)
%GETSNPSDAREA Summary of this function goes here
%   Detailed explanation goes here
dims=size(Y);
Y=reshape(Y,[dims(1:end-1), dims(end)]);
MSE=0;

D=size(Y,1);
for kk=1:D
     [ sn, ~ ] = GetSnPSD(Y(kk,:));
     MSE=MSE+sn^2/D;
end


end


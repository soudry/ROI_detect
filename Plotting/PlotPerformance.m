function PlotPerformance(performance)
% input
v2struct(performance)
% performance - a struct containing information on the performance of the FISTA algorithm:
%       MSE_array - mse error (residual): ||x-data||^2, at each itetation of FISTA
%       MSE_top and MSE_bottom - naive bounds on the MSE error
%       sparsity_ratio - fraction pixels in x which are non zero (at least for some t), at each itetation of FISTA
%       Error_array - the FISTA objective, at each itetation of FISTA
%       t_elapsed - simulation time in sec

%% performance
figure(5631)
subplot(3,1,1)
plot(Error_array);
title('Total Error');
subplot(3,1,2)
hold off
plot(MSE_array,'b');
hold all
plot(0*MSE_array+MSE_top,'g');
hold all
plot(0*MSE_array+MSE_bottom,'r');
hold all
legend('MSE','upper bound','lower bound');%,'Eftychios lower bound');
subplot(3,1,3)
plot(sparsity_ratio);
title('sparsity ratio');  

end


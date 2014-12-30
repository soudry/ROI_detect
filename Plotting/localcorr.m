function rho = localcorr( data )
%localcorr get local correlation of the calcium imaging data
%Input:
%data   M x N x T movie
%
%Output:
%rho    M x N matrix, cross-correlation with adjacent pixel
%
%Author: Yuanjun Gao

[M, N, T] = size(data);
rho = zeros(M, N);

%normalizing data
data = bsxfun(@minus, data, mean(data, 3));
dataStd = sqrt(mean(data.^2, 3));
data = bsxfun(@rdivide, data, dataStd);

%calculate cross-correlation
rho1 = mean(data(1:(end - 1), :, :) .* data(2:end, :, :), 3);
rho2 = mean(data(:, 1:(end - 1), :) .* data(:, 2:end, :), 3);

%add all cross-correlation
rho(1:(end - 1), :) = rho(1:(end - 1), :) + rho1;
rho(2:end, :) = rho(2:end, :) + rho1;
rho(:, 1:(end - 1)) = rho(:, 1:(end - 1)) + rho2;
rho(:, 2:end) = rho(:, 2:end) + rho2;

%normalize by number of adjacent pixels
nNgbr = 4 + zeros(M, N);
nNgbr([1,end], :) = nNgbr([1,end], :) - 1;
nNgbr(:, [1,end]) = nNgbr(:, [1,end]) - 1;
rho = rho ./ nNgbr;
 
rho(isnan(rho))=0;

end


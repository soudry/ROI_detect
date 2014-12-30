function res = RegionInsert(X,box,dims)
%CUTREGION Summary of this function goes here
% inputs:
%  dims -  dimensions of result
%  box - Dx2 array defining spatial box to put X in
%  X -  Input matrix (prod(diff(box,2))xT)

% outputs:
%  res - a [XxYxZ...]xT array

%% 

D=length(dims)-1;

ind=ones(dims(1:(end-1)));
R=1;
for dd=1:D
   d_ind=([1:dims(dd)]>max(box(dd,2))); %#ok
   eval(['ind(' repmat(':,',[1 (dd-1)]) 'd_ind,' repmat(':,',[1 (D-dd)]) ':)=0;']);
   d_ind=([1:dims(dd)]<min(box(dd,1))); %#ok
   eval(['ind(' repmat(':,',[1 (dd-1)]) 'd_ind,' repmat(':,',[1 (D-dd)]) ':)=0;'])
   R=R*(box(dd,2)-box(dd,1)+1);
end


ind_array=repmat(ind,[ones(1,D), dims(end)]);
res=zeros(dims);
res(ind_array>0.5)=reshape(X,[R,dims(end)]);

% func=@(A,B) A(B);
% res=zeros(dims);
% bsxfun(func,res,ind)=reshape(X,[R,dims(end)]);


%%

end


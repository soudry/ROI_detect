function res=AvgData(data,K)
dims=size(data);
D=length(dims);

if K>dims(end)
    error('K must be larger then duration of data');
end

%average over time
if ~isempty(K)
    N=floor(dims(end)/K);
    temp=zeros([dims(1:end-1),K]);
    for tt=1:K-1            
        indices=(1:N)+(tt-1)*N;
        if D==3
            temp(:,:,tt)=mean(data(:,:,indices),3);            
        else
           temp(:,:,:,tt)=mean(data(:,:,:,indices),4);                  
        end
    end
    reminder=mod(dims(end),K);
    if reminder==0
        reminder=N;
    end
    indices=(1:reminder)+N*(K-1);
    if D==3
        temp(:,:,end)=mean(data(:,:,indices),3);
    else
        temp(:,:,:,end)=mean(data(:,:,:,indices),4);
    end
    res=temp;
end
function VideoResidual(data,estimate,file_name,varargin)
%MAKE_VIDEO Summary of this function goes here
% make video file comparing data and estimate
% inputs:
% data - 3D or 4D array
% estimate - 3D or 4D array, same size as data
% file_name - string, if empty, don't save
% varagin - can include u, a bias term
fontsize=15;
FrameRate=10; %Hz
save_video=~isempty(file_name);
if save_video
    homefolder = GetHomeFolder;
    fullpath=fullfile( homefolder, ['NMF_Residual_Video_' file_name]);
end

fh=figure(203);

dims=size(data);
is3D=length(dims)>3;
% bias=0*data;

if ~isempty(varargin)
%     u=varargin{1};
%     f=varargin{2};
%     bias=bsxfun(@times,u,f);
    bias=varargin{1};
else
%     u=quantile(data,0.2,length(dims));
    u=mean(data,3);
    bias=repmat(u,[ones(1,length(dims)-1) dims(end)]);
%     bias=0*data;
end


% ma2=max(data(:));
% mi2=min(data(:));


if is3D
    for kk=1:dims(3)
        if save_video
            if exist('writerObj','var')
                close(writerObj)
            end
            set(gcf,'units','normalized','outerposition',[0 0 1 1])
            set(gcf,'Renderer','zbuffer');
            writerObj = VideoWriter([ fullpath '_z=' num2str(kk)],'MPEG-4'); %
            writerObj.FrameRate=10;
            open(writerObj);
        end
        temp=data(:,:,kk,:);
        ma=max(temp(:));
        mi=min(temp(:));%min(abs(Neuron_spatial(:)))*min(abs(mean(activity,2)));
        temp=bias(:,:,kk,:);
        ma2=max(temp(:));
        mi2=min(temp(:));
        temp=estimate(:,:,kk,:);
        ma3=max(temp(:));
        mi3=min(temp(:));
        temp=(data(:,:,kk,:)-bias(:,:,kk,:)-estimate(:,:,kk,:));
        ma4=max(temp(:));
        mi4=min(temp(:));

        for tt=1:dims(end)
            subplot(2,2,1)
            imagesc(data(:,:,kk,tt),[mi ma]);     
%             title(['Data z=' num2str(kk)],'fontsize',fontsize);
            title(['Data z=' num2str(kk) ' t=' num2str(tt) '/' num2str(dims(end)) ] ,'fontsize',fontsize);
            colorbar
            subplot(2,2,2)
            imagesc(bias(:,:,kk,tt),[mi2 ma2+eps]);     
            title(['Background z=' num2str(kk)],'fontsize',fontsize);
            colorbar
            subplot(2,2,3)
            imagesc(estimate(:,:,kk,tt),[mi3 ma3]); 
            title(['Neurons z=' num2str(kk)],'fontsize',fontsize);
            colorbar
            subplot(2,2,4)
            imagesc(data(:,:,kk,tt)-bias(:,:,kk,tt)-estimate(:,:,kk,tt),[mi4 ma4]); %colorbar
            title(['Residual x4 z=' num2str(kk)],'fontsize',fontsize);
            colorbar
            if save_video
                frame=getframe(fh);
                writeVideo(writerObj,frame);
                disp(['t=' num2str(tt) '/' num2str(dims(end))])
            else
            pause(0.1);    
        end 
        end
    end
else

        ma=max(data(:));
        mi=min(data(:));%min(abs(Neuron_spatial(:)))*min(abs(mean(activity,2)));
        ma2=max(bias(:));
        mi2=min(bias(:));
        ma3=max(estimate(:));
        mi3=min(estimate(:));
        temp=(data-bias-estimate);
        ma4=max(temp(:));
        mi4=min(temp(:));
        set(gcf,'units','normalized','outerposition',[0 0 1 1])
    if save_video
        if exist('writerObj','var')
            close(writerObj)
        end
        set(gcf,'Renderer','zbuffer');
        writerObj = VideoWriter(fullpath,'MPEG-4'); %
        writerObj.FrameRate=FrameRate;
        open(writerObj);
    end
    for tt=1:dims(end)
            subplot(2,2,1)
            imagesc(data(:,:,tt),[mi ma]);     
            colorbar
            title(['Data t=' num2str(tt) '/' num2str(dims(end)) ] ,'fontsize',fontsize);
            subplot(2,2,2)
            imagesc(bias(:,:,tt),[mi2 ma2]);     
            colorbar
            title('Background','fontsize',fontsize);
            subplot(2,2,3)
            imagesc(estimate(:,:,tt),[mi3 ma3]); 
            colorbar
            title('Neurons','fontsize',fontsize);
            subplot(2,2,4)
            res=data(:,:,tt)-bias(:,:,tt)-estimate(:,:,tt);
            imagesc(res,[mi4 ma4]); %colorbar
            colorbar
            title('Residual','fontsize',fontsize);
        if save_video
            frame=getframe(fh);
            writeVideo(writerObj,frame);
            disp(['t=' num2str(tt) '/' num2str(dims(end))])
        else
            pause(0.1);    
        end 
    end
end

if save_video
    close(writerObj)
end



end


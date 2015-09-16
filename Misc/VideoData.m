function VideoData(data,save_video)
if ndims(data)>3
    kk=randi(size(data,3));
    data=data(:,:,kk,:);
end

fontsize=15;
FrameRate=10; %Hz
if save_video
    homefolder = GetHomeFolder;
    fullpath=fullfile( homefolder, 'Data', 'Videos', ['Data' ]);
end

fh=figure(1)

ma=max(data(:));
mi=min(data(:));%min(abs(Neuron_spatial(:)))*min(abs(mean(activity,2)));
dims=size(data);

if save_video
    if exist('writerObj','var')
        close(writerObj)
    end
%     set(gcf,'units','normalized','outerposition',[0 0 1 1])
    set(gcf,'Renderer','zbuffer');
    writerObj = VideoWriter(fullpath,'MPEG-4'); %
    writerObj.FrameRate=FrameRate;
    open(writerObj);
end

for tt=1:dims(end)
    imagesc(data(:,:,tt),[mi ma]);  
    title(['frame=' num2str(tt) '/' num2str(dims(end))],'fontsize',fontsize); 
    if save_video
        frame=getframe(fh);
        writeVideo(writerObj,frame);
        disp(['t=' num2str(tt) '/' num2str(dims(end))])
    else
        pause(0.1);    
    end 
end



end


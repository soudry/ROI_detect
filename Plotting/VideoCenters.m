function VideoCenters(data,neuron_stats,file_name,show_contours)
%MAKE_VIDEO Summary of this function goes here
% make video file comparing data and estimate
% inputs:
% data - 3D or 4D array
% neuron_stats - centers, ROIs and activity
% file_name - string, if empty, don't save
% show_contours - should we show also neuronal contours

v2struct(neuron_stats);
L=size(cent,1);
% for ll=1:L
%     mi=min(activity(ll,:));ma=max(activity(ll,:));
%     activity(ll,:)=(activity(ll,:)-mi)/(ma-mi);
% end


dims=size(data);
FrameRate=10; %Hz
save_video=~isempty(file_name);
if save_video
    homefolder = GetHomeFolder;
    fullpath=fullfile( homefolder, 'FISTAResults','Video', ['Centers' file_name]);
end
is3D=length(dims)>3;

%% Make video with centers and contours
fh=figure(2000);
set(fh,'units','normalized','outerposition',[0 0 1 1])
max_data=max(data(:));
min_data=min(data(:));


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



if is3D
    temp=divisor(dims(3));
    a=max(temp(temp<=sqrt(dims(3))));
    b=dims(3)/a;
    ha = tight_subplot(a,b,[.01 .01],[.01 .01],[.01 .01]);
    for kk=1:dims(3)
        axes(ha(kk)); %#ok
        set(gca,'XTickLabel','','YTickLabel','');
        grid on;
    end
end

for ii=1:dims(end)
        
    if is3D
         for kk=1:dims(3)
            axes(ha(kk)); %#ok
            imagesc(data(:,:,kk,ii),[min_data max_data])
            ind=find(cent(:,3)==kk);
%             hold all; scatter(cent(ind,2),cent(ind,1),ceil(100*activity(ind,ii).^1.5+eps),'w');
            hold all; scatter(cent(ind,2),cent(ind,1),20,'ow');
            hold all; scatter(cent(ind,2),cent(ind,1),20,'xk');
            if show_contours
                hold all; contour(ROI(:,:,kk),L,'k');            
            end
            title(['t=' num2str(ii) '/' num2str(dims(end))])
        end
    else
        imagesc(data(:,:,ii),[min_data,max_data])
        hold all; scatter(cent(:,2),cent(:,1),20,'ow');    
        hold all; scatter(cent(:,2),cent(:,1),20,'xk');
        title(['t=' num2str(ii) '/' num2str(dims(end))])
    %     hold all; contour(neurons_ROI,L,'k' );
        if show_contours
        hold all; contour(ROI,L,'k'); 
        end
%         if show_ground_truth
%             hold all; scatter(real_cent(:,2),real_cent(:,1),3*specs.params.neuron_size,'xg');
%         end
        hold off
    end 
        if save_video
            frame=getframe(fh);
            writeVideo(writerObj,frame);            
        else
            pause(0.01);    
        end
end

if save_video
    close(writerObj)
end


end


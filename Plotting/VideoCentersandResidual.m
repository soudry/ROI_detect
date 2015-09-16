function VideoCentersandResidual(data,shapes,activity,box_cell,cent,ranks,estimate,file_name,varargin)
%MAKE_VIDEO Summary of this function goes here
% make video file comparing data and estimate
% inputs:
% data - 3D or 4D array
% shapes , activity, box_cell, cent, ranks
% file_name - string, if empty, don't save
% varagin - can include a bias term

subplot = @(m,n,p) subtightplot (m, n, p, [0.03 0.02], [0.03 0.05], [0.03 0.01]);
fontsize=15;
FrameRate=10; %Hz
edge_distance=0; % safe distance from edge
quality=62; %movie quality (1-100, bad to high quality, and small to large file size)
a=6; % number of rows
b=9; % number of columns
c=3; % big patch width

% activity_mat=cell2mat(activity);
% activity_mat(activity_mat<0)=0;
% activity_mat=activity_mat+eps;
% activity_mat=20*bsxfun(@times,activity_mat,1./mean(activity_mat,2));

save_video=~isempty(file_name);
if save_video
    homefolder = GetHomeFolder;
    fullpath=fullfile( homefolder, ['NMF_Video_' file_name]);
end

fh=figure(2);
set(fh,'units','normalized','outerposition',[0 0 1 1]);
% setappdata(gcf, 'SubplotDefaultAxesLocation', [0,0,1,1]) %make smal margins
dims=size(data);
S_dims=dims;
S_dims(end)=1;
is3D=length(dims)>3;
% bias=0*data;

if ~isempty(varargin)
    bias=varargin{1};
else
%     u=dims(end)*quantile(data,0.2,length(dims));
    u=sum(data,3);
    bias=repmat(u/dims(end),[ones(1,length(dims)-1) dims(end)]);
%     bias=0*data;
end

%% scale factors
scale=1; %zoom intensity scale
ma=max(data(:));
mi=min(data(:));%min(abs(Neuron_spatial(:)))*min(abs(mean(activity,2)));
ma2=max(bias(:))+eps;
mi2=min(bias(:));
ma3=max(data(:));
mi3=min(data(:));
mi4=min(data(:)-bias(:)-estimate(:));
ma4=max(data(:)-bias(:)-estimate(:));

% mi_pp=zeros(4); ma_pp=zeros(4);
% for pp=1:length(ranks)                    
%     index=ranks(pp);    
%     activity_pp=activity{index};
%     shapes_pp=shapes{index};
%     mi_pp(pp)=min(shapes_pp(:))*min(activity_pp(:));
%     ma_pp(pp)=max(shapes_pp(:))*max(activity_pp(:));
% end

    if save_video
        if exist('writerObj','var')
            close(writerObj)
        end
        set(gcf,'units','normalized','outerposition',[0 0 1 1])
        set(gcf,'Renderer','zbuffer');
        writerObj = VideoWriter(fullpath,'MPEG-4');
        writerObj.FrameRate=FrameRate;
        writerObj.Quality=quality;
        open(writerObj);
    end
    

    
    
    for tt=1:dims(end)
            subplot(a,b,[1:c, (1:c)+b, (1:c)+2*b])
            hold off; imagesc(data(:,:,tt),[mi ma]);
%             if tt==1
                colorbar
                title('Raw data','fontsize',fontsize);
%                 set(gca,'xtick',[],'ytick',[]);
%             end
            hold all; 
%             scatter(cent(2,:),cent(1,:),activity_mat(tt,:),'wo');
%             hold all; scatter(cent(2,:),cent(1,:),activity_mat(tt,:),'kx');%        
            k=1:length(cent(2,:)); text(cent(2,:),cent(1,:),num2str(k'),'color','w')
            xlim([edge_distance dims(2)-edge_distance]);
            ylim([edge_distance dims(1)-edge_distance]);
            
            subplot(a,b,c*b+[1:c, (1:c)+b, (1:c)+2*b])
            imagesc(bias(:,:,tt),[mi2 ma2]);
            colorbar
            title('Background','fontsize',fontsize);
%             set(gca,'xtick',[],'ytick',[]);            
             hold all; 
%               scatter(cent(2,:),cent(1,:),max(activity_mat(:,:),[],1),'wo');
%             hold all; scatter(cent(2,:),cent(1,:),max(activity_mat(:,:),[],1),'kx');   
            k=1:length(cent(2,:)); text(cent(2,:),cent(1,:),num2str(k'), 'color','w')
            xlim([edge_distance dims(2)-edge_distance]);
            ylim([edge_distance dims(1)-edge_distance]);
            
            subplot(a,b,c+[1:c, (1:c)+b, (1:c)+2*b])
            hold off;  imagesc(estimate(:,:,tt),[mi3, mi3+(ma3-mi3)*scale]); 
%             if tt==1
                colorbar
                title(['Denoised - Timestep ' num2str(tt) '/' num2str(dims(end))],'fontsize',fontsize);
%                 set(gca,'xtick',[],'ytick',[]);
%             end            
            hold all; 
%             scatter(cent(2,:),cent(1,:),activity_mat(tt,:),'wo');
%             hold all; scatter(cent(2,:),cent(1,:),activity_mat(tt,:),'kx');           
            k=1:length(cent(2,:)); text(cent(2,:),cent(1,:),num2str(k'),'color','w')
            
            xlim([edge_distance dims(2)-edge_distance]);
            ylim([edge_distance dims(1)-edge_distance]);
            subplot(a,b,2*c+[1:c, (1:c)+b, (1:c)+2*b])
            res=data(:,:,tt)-bias(:,:,tt)-estimate(:,:,tt);
            hold off; imagesc(res,[mi4, mi4+scale*(ma4-mi4)]); %colorbar
%             if tt==1
                colorbar
                title(['Residual'],'fontsize',fontsize);
%                 set(gca,'xtick',[],'ytick',[]);
%             end
            hold all; 
%             scatter(cent(2,:),cent(1,:),activity_mat(tt,:),'wo');
%             hold all; scatter(cent(2,:),cent(1,:),activity_mat(tt,:),'kx');    
            k=1:length(cent(2,:)); text(cent(2,:),cent(1,:),num2str(k'),'color','k')
            
            xlim([edge_distance dims(2)-edge_distance]);
            ylim([edge_distance dims(1)-edge_distance]);
            
            
            for pp=1:length(ranks)
                subplot(a,b,b*c+c+pp)
                index=ranks(pp);
                area=box_cell{index};
                box_data=data(area(1,1):area(1,2),area(2,1):area(2,2),:)-bias(area(1,1):area(1,2),area(2,1):area(2,2),:);
                mi_box=0;%min(box_data(:));
                ma_box=max(box_data(:));
                activity_pp=activity{index};
                shapes_pp=shapes{index};
                mat=RegionInsert(shapes_pp*activity_pp(tt),area,S_dims);
%                 imagesc(mat,[mi_pp(pp) scale*ma_pp(pp)]);
                imagesc(mat,[mi_box scale*ma_box]);
                ylim([area(1,1) area(1,2)]);
                xlim([area(2,1) area(2,2)]);
                text(area(2,1)+2,area(1,1)+5,num2str(pp),'color','w','fontweight','bold')

%                 set(gca,'xtick',[],'ytick',[],'DataAspectRatio',[1 1 1]);
                if pp==3    
                    title('Representative spatial components', 'Units', 'normalized', ...
                        'Position', [0.1 1],'HorizontalAlignment', 'left','fontsize',fontsize);
                end
                colorbar('location','EastOutside');
                subplot(a,b,b+b*c+c+pp)
                imagesc(data(:,:,tt)-bias(:,:,tt),[mi_box scale*ma_box]);
                ylim([area(1,1) area(1,2)]);
                xlim([area(2,1) area(2,2)]);
                text(area(2,1)+2,area(1,1)+5,num2str(pp),'color','w','fontweight','bold')

%                 set(gca,'xtick',[],'ytick',[],'DataAspectRatio',[1 1 1]); 
                if pp==3    
                    title('Centralized data patches', 'Units', 'normalized', ...
                        'Position', [0.5 1],'HorizontalAlignment', 'left','fontsize',fontsize);
                end
                colorbar('location','EastOutside');
                subplot(a,b,2*b+b*c+c+pp)
                imagesc(data(:,:,tt),[mi ma]);
                ylim([area(1,1) area(1,2)]);
                xlim([area(2,1) area(2,2)]);
                text(area(2,1)+2,area(1,1)+5,num2str(pp),'color','w','fontweight','bold')

%                 set(gca,'xtick',[],'ytick',[],'DataAspectRatio',[1 1 1]); 
                if pp==3    
                    title('Raw data patches', 'Units', 'normalized', ...
                        'Position', [0.7 1],'HorizontalAlignment', 'left','fontsize',fontsize);
                end
                colorbar('location','EastOutside');
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


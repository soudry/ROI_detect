function PlotEachNeuron(data,v_s,v_t,box_cell,varargin)
% Inputs:
% data - XxYxT data array
% v_s - cell array of L neuronal 'shape' vectors (length XY or shorter)
% v_t - cell array of L neuronal 'activity' vectors (length T)
% box_cell - cell array of L 2x2 matrices with the edges of the box corresponding to each neuron
% file_name - currently not working (for later version, if we want to save figures)

L=length(v_s);
dims=size(data);
S_dims=dims;
S_dims(end)=1;
% temp=divisor(L);
% a=max(temp(temp<=sqrt(L)));
% b=L/a;

% set(h1,'units','normalized','outerposition',[0 0 1 1])

% pic=localcorr( data );
pic=mean( data ,ndims(data));
prev_key_pressed=[];
scale=1;

plot_centers=0;
if length(varargin)==1
    cent=varargin{1};
    plot_centers=1;
end

for ll=1:L
    h=figure(525)
    set(gcf,'units','normalized','outerposition',[0 0 1 1])
    area=box_cell{ll};
    subplot(3,3,[4 5 6 7 8 9])
    
    if dims>3
        box=box_cell{ll};
        zz=round((area(3,1)+area(3,2))/2)
        imagesc(pic(:,:,zz))
    else
        imagesc(pic)
    end
    hold all
%     if ll>1
%         set(rh,'linewidth',1,'linestyle','--');
%     end
    rh=rectangle('Position',[area(2,1),area(1,1),area(2,2)-area(2,1),area(1,2)-area(1,1)],'linewidth',2);    
    if plot_centers        
        hold all; scatter(cent(2,1:ll),cent(1,1:ll),2,'ow');    
        hold all; scatter(cent(2,1:ll),cent(1,1:ll),2,'xk');
    end

    xlabel('press "n" to show next neuron');
    
    subplot(3,3,1)        
    mat=RegionInsert(abs(v_s{ll}),area,S_dims);
    if ndims(mat)==3
        mat=mean(mat,3); %collapse 3D shape to 1D one
    end
    imagesc(mat);
    ylim([area(1,1) area(1,2)]);
    xlim([area(2,1) area(2,2)]);
    title(['neuron# ' num2str(ll)])
    
    subplot(3,3,3)            
    plot(v_t{ll},'k','linewidth',0.5)
    
    key_pressed=[];
    hp=[];
    
    if ndims(mat)==2
        data_box=data(area(1,1):area(1,2),area(2,1):area(2,2),:);
    else
        data_box=squeeze(data(area(1,1):area(1,2),area(2,1):area(2,2),zz,:));
    end
    mi=min(data_box(:));
    ma_abs=max(data_box(:));    
    tt=0;
    
        while tt<dims(end)
            tt=tt+1;
            ma=mi+scale*(ma_abs-mi);
            subplot(3,3,2)
            imagesc(data(:,:,tt),[mi ma]);
            ylim([area(1,1) area(1,2)]);
            xlim([area(2,1) area(2,2)]); 
%             mag=sum(v_s{ll}.^2)*sum(v_t{ll}.^2);
            mag=sum(v_s{ll})*max(v_t{ll});
            title(['Magnitude=' num2str(mag,2) ])
            subplot(3,3,3)
            hold all
            temp=v_t{ll};
            delete(hp);
            hp=plot(temp(1:tt),'r','linewidth',3);          
            title(['t=' num2str(tt) '/' num2str(dims(end))])
            
            % stop if 's' is pressed, and pause if 'p' is pressed            
            key_pressed=get(h,'currentkey');
            pause(0.01);
            if ~strcmp(key_pressed,prev_key_pressed)
                prev_key_pressed=key_pressed;
                switch  key_pressed
                    case 'n' %next neuron
                     tt=dims(end);
                    case '9'
                    jump_forward=100;
                    tt=tt+jump_forward;
                    case '0'
                    jump_forward=100;
                    tt=tt+jump_forward;                    
                    case '1'
                    jump_backward=500;
                    tt=max(tt-jump_backward,1);
                    case '2'
                    jump_backward=200;
                    tt=max(tt-jump_backward,1);
                    case 'i'
                    scale=scale*2;
                    disp('video scale increase');                    
                    case 'd'
                    scale=scale/2;
                    disp('video scale decreased');                    
                end
                figure(h)
            end
        end        
    close(h)  
end
% 
% if ~isempty(file_name)
%     homefolder = GetHomeFolder;
%     folder=fullfile( homefolder, 'Data', 'Neurons');
%     figure_name=[ file_name '.png'];
%     figure(1)
%     Export2Folder([ 'shapes_' figure_name],folder);
% end

end


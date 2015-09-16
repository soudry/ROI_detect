function PlotCenters(data,cent)
%PLOTCENTERS plots cent (infered neuronal centers) on top of data

L=size(cent,1);
R_list=ones(L,2);

is3D=(ndims(data)>3);
if is3D
    num_figs=size(data,3);
else
    num_figs=1;
end

    a=floor(sqrt(num_figs)); b=ceil(sqrt(num_figs));
show_centers=~isempty(cent); %
ck=1:length(cent(:,2));
fh=figure;
colormap('paruly')

for zz=1:num_figs
%     fh=figure(zz);
    set(fh,'units','normalized','outerposition',[0 0 1 1]);
%     title('top 5% quantile image')
    
    if is3D
        Y=squeeze(data(:,:,zz,:));
    else
        Y=data;
    end
%     
%     subplot(2,2,1)
%     imagesc(mean(abs(Y),3))
%     title('mean image')
%     subplot(2,2,2)
%     imagesc(max(abs(Y),[],3))
%     title('max image')
%     subplot(2,2,3)
%     imagesc(std(abs(Y),[],3))
%     title('std image')    
%     subplot(2,2,4)
%     imagesc(localcorr( Y ));
%     title('correlation image')
%     
    subplot(a,b,zz)
    imagesc(quantile( Y,0.95 ,3));
    title(['z=' num2str(zz)])
end

for zz=1:num_figs
    if show_centers
%         for ii=1:4
%             subplot(2,2,ii)
            if is3D
                subplot(a,b,zz)
                ind=find(cent(:,3)==zz);
%                 if zz==num_fig
%                     ind_down=find(cent(:,3)==zz-1);
%                     ind_up=[];
%                 elseif zz==1
%                     ind_up=find(cent(:,3)==zz+1);
%                     ind_down=[];
%                 else
%                     ind_up=find(cent(:,3)==zz+1);
%                     ind_down=find((cent(:,3)==zz-1));
%                 end
                hold all; scatter(cent(ind,2),cent(ind,1),20,'ow');    
                hold all; scatter(cent(ind,2),cent(ind,1),20,'xk');
                text(1.01*cent(ind,2),1.01*cent(ind,1),num2str(ck(ind)'),'color','r')
                
                for kk=1:length(ind)
                    ll=ind(kk);
                    x=cent(ll,2)-R_list(ll,2);
                    y=cent(ll,1)-R_list(ll,1);
                    w=2*R_list(ll,2);
                    h=2*R_list(ll,1);
                    rectangle('Position',[x y w h]);
                    if zz<num_figs
%                     figure(zz+1)
                    subplot(a,b,zz+1)
%                     subplot(2,2,ii)
                    rectangle('Position',[x y w h],'linestyle','--');
                    end
                    if zz>1
%                     figure(zz-1)
                    subplot(a,b,zz-1)
%                     subplot(2,2,ii)
                    rectangle('Position',[x y w h],'linestyle','--');
                    end
                    subplot(a,b,zz)
%                     subplot(2,2,ii)
                end
                
%                 for kk=1:length(ind_up)
%                     ll=ind_up(kk);
%                     x=cent(ll,2)-R_list(ll,2);
%                     y=cent(ll,1)-R_list(ll,1);
%                     w=2*R_list(ll,2);
%                     h=2*R_list(ll,1);
%                     rectangle('Position',[x y w h]);                    
%                 end
%                 
                
                
            else
                hold all; scatter(cent(:,2),cent(:,1),20,'ow');    
                hold all; scatter(cent(:,2),cent(:,1),20,'xk');
                ck=1:L; text(cent(:,2),cent(:,1),num2str(ck'),'color','w')
                for ll=1:L
                    x=cent(ll,2)-R_list(ll,2);
                    y=cent(ll,1)-R_list(ll,1);
                    w=2*R_list(ll,2);
                    h=2*R_list(ll,1);
                    rectangle('Position',[x y w h]);
                end
            end
%         end
    end
end


end


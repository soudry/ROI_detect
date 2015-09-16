function [ha1, ha2]=PlotAllNeurons(v_s_lp,v_t,box_cell,file_name)

L=length(v_t);
%% determinse size and ordering of subplots
max_plots_in_figure=35;
max_plots_in_row=5;
if L<=max_plots_in_figure
    temp=divisor(L);
    a=max(temp(temp<=sqrt(L))); %rows
    b=L/a; %colmuns
    if b>max_plots_in_row
        b=max_plots_in_row;
        a=ceil(L/b);
    end
else
    b=max_plots_in_row;
    a=ceil(max_plots_in_figure/b);    
end
%%

if ~isempty(file_name)
    folder = GetHomeFolder;
    figure_name=[ file_name '.png'];
end


if size(box_cell{1},1)>2
    Z=box_cell{1}(3,2);
else
    Z=1
end

    
fig_count=0;
for ll=1:L
        if  ~mod(ll-1,max_plots_in_figure)
            if and(ll~=1,~isempty(file_name))
                figure(h2)
                Export2Folder([ 'activity' num2str(fig_count) '_' figure_name],folder);     
            end
            h2=figure(201+fig_count);
            ha2 = tight_subplot(a,b,[.05 .03],[.03 .05],[.03 .01]) ;
            fig_count=fig_count+1;
        end
        subplot_ind=mod(ll-1,max_plots_in_figure)+1;
        set(h2,'units','normalized','outerposition',[0 0 1 1])
    
        axes(ha2(subplot_ind));
%         subplot(a,b,subplot_ind)
        plot(v_t{ll}*sum(v_s_lp{ll}));

%         set(gca,'Xlim',[0 length(v_t{ll})],'Xtick',[],'Ytick',[],'Xticklabel',[],'Yticklabel',[])
        set(gca,'Xlim')
        if ll<L
            title(['neuron #' num2str(ll)])  
        else
            title([' Background'])  
        end
        
end

if ~isempty(file_name)
    figure(h2)
    Export2Folder([ 'activity' num2str(fig_count) '_' figure_name],folder);     
end


for zz=1:Z
    fig_count=0;
    for ll=1:L
        if  ~mod(ll-1,max_plots_in_figure)
            if  and(ll~=1,~isempty(file_name))
                figure(h1)
                if Z>1
                    Export2Folder([ 'shapes' num2str(fig_count)  '_z=' num2str(zz) '_' figure_name],folder);
                else
                    Export2Folder([ 'shapes' num2str(fig_count)  '_' figure_name],folder);
                end
            end
            h1=figure(110+fig_count+zz*100)
            set(h1,'units','normalized','outerposition',[0 0 1 1])
            ha1 =  tight_subplot(a,b,[.05 .03],[.03 .05],[.03 .01]) ;
            fig_count=fig_count+1;
         end
    
        
        subplot_ind=mod(ll-1,max_plots_in_figure)+1;

        axes(ha1(subplot_ind));
 

        mat = Vector2Mat(v_s_lp{ll},box_cell{ll});

    
        mi=0; ma=max(mat(:));
        current_box=box_cell{ll}
        if Z>1
        if and(zz>=current_box(3,1),zz<=current_box(3,2))
            imagesc(squeeze(mat(:,:,-current_box(3,1)+1+zz)),[mi ma]);
        else
            imagesc(0*squeeze(mat(:,:,1)),[mi ma]);
        end
        else
            imagesc(mat,[mi ma]);
        end
%         set(gca,'Xtick',[],'Ytick',[],'Xticklabel',[],'Yticklabel',[])
        
        if Z>1
            if ll<L
                title(['neuron #' num2str(ll) '_z=' num2str(zz)])
            else
                title([' Background_z=' num2str(zz)])
            end
        else            
            if ll<L
                title(['neuron #' num2str(ll)])  
            else
                title([' Background'])  
            end            
        end
    end

if ~isempty(file_name)
    figure(h1)
    if Z>1
        Export2Folder([ 'shapes'  num2str(fig_count) '_z=' num2str(zz) '_' figure_name],folder);
    else
        Export2Folder([ 'shapes' num2str(fig_count)  '_' figure_name],folder);
    end
end


    
    

end


function [ha1 ha2]=PlotAllNeurons(v_s_lp,v_t,box_cell,file_name)

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
prev_num_figure=-1;
    for ll=1:L
        num_figure=floor((ll-0.5)/max_plots_in_figure);
        subplot_ind=mod(ll-1,max_plots_in_figure)+1;
        
        h1=figure(1+num_figure*2)
        set(h1,'units','normalized','outerposition',[0 0 1 1])
%         title('neurons')
        if prev_num_figure~=num_figure
            h1=figure(1+num_figure*2)
            ha1 = tight_subplot(a,b,[.05 .01],[.01 .05],[.01 .01])
            h2=figure(2+num_figure*2)
            prev_num_figure=num_figure;
            ha2 = tight_subplot(a,b,[.05 .01],[.01 .05],[.01 .01]) 
            h1=figure(1+num_figure*2)
        end
        axes(ha1(subplot_ind));
%         subplot(a,b,subplot_ind)        

        mat = Vector2Mat(v_s_lp{ll},box_cell{ll});
%         [~, cc_num]=bwlabeln(mat>0.5*max(mat(:)));
%        if cc_num>2
%            continue
%        end
        
        imagesc(mat);
        set(gca,'Xtick',[],'Ytick',[],'Xticklabel',[],'Yticklabel',[])
%         ylim([area(1,1) area(1,2)]);
%         xlim([area(2,1) area(2,2)]);    
        h2=figure(2+num_figure*2)
        set(h2,'units','normalized','outerposition',[0 0 1 1])
%         title('activity')  
        axes(ha2(subplot_ind));
%         subplot(a,b,subplot_ind)
        plot(v_t{ll}*sum(v_s_lp{ll}));
        set(gca,'Xlim',[0 length(v_t{ll})],'Xtick',[],'Ytick',[],'Xticklabel',[],'Yticklabel',[])
    end

if ~isempty(file_name)
    homefolder = GetHomeFolder;
    folder=fullfile( homefolder, 'Data', 'Neurons');
    figure_name=[ file_name '.png'];
    figure(1)
    Export2Folder([ 'shapes_' figure_name],folder);
    figure(2)
    Export2Folder([ 'activity_' figure_name],folder);        
end

end


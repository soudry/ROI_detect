function [v_s,v_t] = GreedyNNPCA(data,v_s)
%GREADYNNPCA Summary of this function goes here
% Find bests decomposition of data  array  (XxT or XxYxT or XxYxZxT) to static image v_s
% (Xx1 XxY or XxYxZ) and activity v_t  (Tx1), starting from some initial guess of v_s
% if v_s is empty, then we mean activity as initial guess

nn_activity=1;
greedy_pca_iterations=5;
T_dim=ndims(data);
spatial_dimensions=T_dim-1;
if isempty(v_s)
    v_s=mean(data,T_dim);
end
            
            for ii=1:greedy_pca_iterations
                c_s=sum(v_s(:).^2);
                temp1=bsxfun(@times,data,v_s);
                switch spatial_dimensions
                    case 3
                        temp2=sum(sum(sum(temp1,1),2),3)/c_s;    
                    case 2
                        temp2=sum(sum(temp1,1),2)/c_s;
                    case 1
                        temp2=sum(temp1,1)/c_s;
                end
                
                if nn_activity
                    temp2(temp2<0)=0;
                    v_t=temp2;
                else
                    v_t=temp2; %#ok
                end
                c_t=sum(v_t(:).^2);
                temp3=bsxfun(@times,data,v_t);
                temp4=squeeze(sum(temp3,T_dim))/c_t;
                temp4(temp4<0)=0;
                v_s=temp4;
            end        
%     v_t=squeeze(v_t);
end


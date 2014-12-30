function [v_s,v_t] = GreedyNNPCA(X,v_s)
%GREADYNNPCA Summary of this function goes here
% Find bests decomposition of video X (XxYxT) array to static image v_s
% (XxY) and activity v_t  (Tx1), starting from some initial guess of s
%   Detailed explanation goes here
nn_activity=1;
greedy_pca_iterations=5;
T_dim=ndims(X);
is3D=T_dim==4;
            
            for ii=1:greedy_pca_iterations
                c_s=sum(v_s(:).^2);
                temp1=bsxfun(@times,X,v_s);
                if is3D
                    temp2=sum(sum(sum(temp1,1),2),3)/c_s;    
                else
                    temp2=sum(sum(temp1,1),2)/c_s;
                end
                
                if nn_activity
                    temp2(temp2<0)=0;
                    v_t=temp2;
                else
                    v_t=temp2; %#ok
                end
                c_t=sum(v_t(:).^2);
                temp3=bsxfun(@times,X,v_t);
                temp4=squeeze(sum(temp3,T_dim))/c_t;
                temp4(temp4<0)=0;
                v_s=temp4;
            end        
%     v_t=squeeze(v_t);
end


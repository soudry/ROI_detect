function cent = GetCenters(pic)
% find x-y locations of maxima in an image 'pic'

centers=imregionalmax(pic);
if all(~pic(:)) %take care of case where pic is all 0's
    centers=centers*0;
end
% centers=imregionalmax(imhmax(pic,median(pic(pic>0)))); 
dims=size(pic);
is3D=length(dims)>2;
if is3D
    L=sum(sum(sum(centers((2:end-1),(2:end-1),:))));
else
    L=sum(sum(centers((2:end-1),(2:end-1))));
end
cent=zeros(L,length(dims));
magnitude=zeros(L,1);
ll=0;

for ii=2:(dims(1)-1) %do not look on the edges
    for jj=2:(dims(2)-1)  %do not look on the edges
        if is3D
            for  kk=1:dims(3)
                if centers(ii,jj,kk)
                    ll=ll+1;            
                    cent(ll,1)=ii;
                    cent(ll,2)=jj;
                    cent(ll,3)=kk;
                    magnitude(ll)=pic(ii,jj,kk);
                end            
            end
        else
            if centers(ii,jj)
                ll=ll+1;            
                cent(ll,1)=ii;
                cent(ll,2)=jj;
                magnitude(ll)=pic(ii,jj);
            end
        end
    end
end

[magnitude, ind]=sort(magnitude,'descend');
cent=cent(ind,:);


end


function box = GetBox(centers,R,dims)
D=length(R);
box=zeros(D,2);

    for dd=1:D
        d_region=centers(dd)+(-R(dd):R(dd));
        d_region(d_region<1)=[]; d_region(d_region>dims(dd))=[]; 
        box(dd,1)=min(d_region); box(dd,2)=max(d_region);
    end

end


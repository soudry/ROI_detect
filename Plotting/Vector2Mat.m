function mat = Vector2Mat(vector,box)
%VECTOR2MAT Summary of this function goes here
%   Detailed explanation goes here

if size(box,1)>2 %3D case
    mat=reshape(vector,[1+box(1,2)-box(1,1),1+box(2,2)-box(2,1),1+box(3,2)-box(3,1)]);
else
    mat=reshape(vector,[1+box(1,2)-box(1,1),1+box(2,2)-box(2,1)]);
end
% mat=reshape(vector,41,41);


end


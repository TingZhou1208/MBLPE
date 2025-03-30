function pos=remove_empty(pos,n_var)
n=size(pos,1);
for i=1:n
    x= pos(i,:);
    x=1.*(x>=0.6); %% 二值化
    if nnz(x)==0
        kk=randi([1,n_var]);
        pos(i,kk)=1;
    end
end
end

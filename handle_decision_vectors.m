function Offspring=handle_decision_vectors(Parent,Offspring,n_var)  %%父代种群，子代种群，函数，训练数据
pp=Parent(:,1:n_var); %%二值化重组种群
sp=Offspring(:,1:n_var);
N1=size(pp,1);

if N1==0
    return;
end

for ii=1:N1
    sp=Offspring(:,1:n_var);  %%二值化种群
    state=check_whether_it_is_shown(pp(ii,:),sp,n_var);
    if size(state,1) ~=0
         Offspring(state,:)=[];     
    end
end
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function index=check_whether_it_is_shown(p_solution,pop_sp,n_var) %%所有后代个体与每个父本进行比较，是否有重复，重复的个体会被删除
u1=p_solution; %%父代个体
n2=size(pop_sp,1);%%后代
index=[];
for ii=1:n2
    if u1(1,1:n_var)==pop_sp(ii,1:n_var)
        index= [index ii];
    end
end
end


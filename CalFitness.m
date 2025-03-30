function [N,R,Fitness] = CalFitness(PopObj,PopDec) %%目标函数值，决策向量
%% Detect the dominance relation between each two solutions---非支配排名
N=size(PopDec,1);
Dominate = false(N);
for i = 1 : N-1
    for j = i+1 : N
        k = any(PopObj(i,:)<PopObj(j,:)) - any(PopObj(i,:)>PopObj(j,:));
        if k == 1
            Dominate(i,j) = true;
        elseif k == -1
            Dominate(j,i) = true;
        end
    end
end

%% Calculate S(i)
S = sum(Dominate,2);
%% Calculate R(i)
R = zeros(1,N);
for i = 1 : N
    R(i) = sum(S(Dominate(:,i)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate D(i)---目标空间距离
Distance = pdist2(PopObj,PopObj);
Distance(logical(eye(length(Distance)))) = inf;
Distance = sort(Distance,2);
D_Obj = 1./(Distance(:,floor(sqrt(N)))+2); %%距离第sqrt(N)个邻居的距离
%%   决策空间距离----改为汉明距离？
Distance = pdist2(PopDec,PopDec,'cityblock');
Distance(logical(eye(length(Distance)))) = inf;
Distance = sort(Distance,2);
D_Dec = 1./(Distance(:,floor(sqrt(N)))+2);
D =  D_Obj+ D_Dec;
%% Calculate the fitnesses
Fitness = R' + D;
end


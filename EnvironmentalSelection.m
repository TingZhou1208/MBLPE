function [Population] = EnvironmentalSelection1127(Population,N,n_var,n_obj)
% The environmental selection of PRDH

%------------------------------- Copyright --------------------------------
% Copyright (c) 2021 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

%% Remove duplicated solutions (search space)
Population=unique(Population,'rows','stable');%%过滤搜索空间掉相同的个体-再一次检查
Obj=Population(:,n_var+1:n_var+n_obj);
Dec=Population(:,1:n_var);
%% 非支配排序获取非支配前沿数和第一层前沿的个体
Front = NDSort(Obj, inf);
Population(:,n_var+n_obj+1)=Front';
MaxF=max(Front);
index = find(Front ==1);
NS=Population(index,:); %%种群中的最优解
NSDec=Dec(index,:);
NSObj=Obj(index,:);
%% 处理目标重复向量
Population1=[];
FrontPop=[];
for i = 2:MaxF
    Next= find(Front==i); %%获取每一层前沿的个体
    PopObj   = Obj(Next,:);
    [UObj,B, c] = unique(PopObj(:,2), 'rows', 'stable');  %%同一层前沿获取所选特征数量相同的个体
    if length(B)==length(Next)    %%判断是否有重复的目标向量
        FrontPop = [FrontPop; Population(Next,:)]; %%表示同一等级没有重复的解
    else
        for j=1:max(c)
            index11 = find(c==j);
            if size(index11, 1) > 1        %%表示有重复的解            PopObj =sortrows(PopObj ,1);
                temp=Population(Next(index11),:);
                tempDec=Dec(Next(index11),:);
                tempObj=Obj(Next(index11),:);
                %% 计算重复解与其目标空间欧式距离较近个体的汉明距离-决策空间-----第一个指标
                poptemp=repmat(tempObj(1,:),size(NSObj,1),1);                %%取出每个个体的的函数值%     SD = pdist2(poptemp, NSObj, 'cityblock');
                distance_Obj=sqrt(sum((poptemp-NSObj).*((poptemp-NSObj)),2));%%根据目标函数值来计算当前个体与种群之间的欧式距离（目标空间）
                [bb,cc]=sort(distance_Obj);                                  %%从小到大排序，输出排序后所对应的个体的索引
                nearest(1,:)=NS(cc(1),:);                                   %%找到领域中距离自己最近的个体
                ds1=[];
                for kk=1:size(tempDec,1)
                    ds1(kk,1)=pdist2(nearest(1,1:n_var), tempDec(kk,1:n_var), 'cityblock')./n_var; %%值越大，不相似程度较大 选取值越大的值
                end
                %% 选取已选特征子集中每维特征被选频数作为当前特征子集选取或者保留的第二个指标-----频数越大，说明该特征一直被选取，意味该特征很重要，保留总频数较大的特征
                Pop=[FrontPop;NS];
                fre=[];          %% 计算每一维特征所选的频数
                for jj=1:n_var
                    fre(1,jj)=sum(Pop(:,jj));
                end
                nu=[];
                for ik=1:size(tempDec,1)
                    index = find(tempDec(ik,:) ==1);
                    nu(ik,1)= sum(fre(index));
                end
                [~,index3] = sortrows([-ds1 nu]); %%拥挤度越大的越好 两个选择标准，第一个标准是局部收敛性质量标准，第二个是拥挤距离指标（）
                selectedNo=temp(index3(1),:);
                FrontPop = [FrontPop; selectedNo];
            else
                FrontPop = [FrontPop;Population(Next(index11),:)];
            end
        end
    end
    Population1=[FrontPop;NS];
end
%% 计算适应度值-再根据适应度值获取排名靠前的个体
[~,~,Fitness]= CalFitness(Population1(:,n_var+1:n_var+n_obj),Population1(:,1:n_var)); %%  方案2 基于epsilon的约束度值+SCD
[~,index]=sort(Fitness,'ascend'); %%升序
Population=Population1(index(1:N),1:n_var+n_obj);
end



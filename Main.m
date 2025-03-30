
addpath(genpath('datasets/'));
addpath(genpath('Data/'));
addpath(genpath('Indicator_calculation/'));
clear all
clc


dataname={
    'Glass';  %%1
    'SRBCT'; %%  13  2308
    };
N_funtion=size(dataname,1);
runtimes=3;
for i_data=2:N_funtion%%17
    %% load data
    s1=dataname{i_data,1};    %%要存储名字中另一个变量 调取第一个数据集
    FileString1= [strcat('train_dataSet_',s1),'.mat']; %%转换数据类型 获取训练集
    FileString2= [strcat('test_dataSet_',s1),'.mat']; %%转换数据类型  获取测试集
    
    load(FileString1,'train_data');
    load(FileString2,'test_data');
    Num_feature=size(train_data,2)-1; %%特征个数，也是编码长度 第1列是数据标签
    %% Initialize parameters in the algorithm
    if Num_feature>300
        popsize=300;
    else
        popsize=Num_feature;
    end
    Max_evaluation=100*popsize;
    Max_Gen=fix(Max_evaluation/popsize);
    fname='FS';  % function name 目标函数
    %% 计算得分
    n_var=Num_feature;
    T = min(n_var, popsize * 3);
    if T < n_var
        fun = @fsFisher;
        feat=train_data(:,2:end);
        label=train_data(:,1);
        [Weight,rkFhData] = fun(feat,label);
        fstSubData = rkFhData(:,1:T);
        subFeat = feat(:,fstSubData);
        
        train_data=[label  subFeat];
        
        n_var=size(subFeat,2);
        testfeat=test_data(:,2:end);
        testlabe=test_data(:,1);
        
        testsubfeat=testfeat(:,fstSubData);
        test_data=[testlabe testsubfeat];
    end
    n_obj=2;
    xl=0*ones(1,n_var);     % the low bounds of the decision variables
    xu=1*ones(1,n_var);     % the up bounds of the decision variables
    repoint=[1,1];          % 参考点设置为[1,1]
    for j=1:runtimes
        %% Search the PSs using NCDE
        %  fprintf('Running test function: %s, times = %d \n', s1,j);
        t1 = cputime;
        [ps,pf]=MBLPE(fname,xl,xu,n_obj,popsize,Max_Gen,train_data);
        t = cputime - t1;
        %% 评估测试集并计算出测试集的HV指标值
        Nsize=size(ps,1);
        fitness_test=ones(Nsize,n_obj);
        for kk=1:Nsize
            fitness_test(kk,:)=Feature_selection_Fun_deal_conti_x(ps(kk,:),train_data,test_data);
        end
        %% Indicators-----评估测试集的指标值
        %  test_HV=Hypervolume_calculation(fitness_test,repoint);
        test_HV=HV(fitness_test,repoint);
        fprintf('Running test function: %s \n %d times test_HV1=%f \n', s1,j,test_HV);
        %% Indicators-----评估训练集的指标值
        %   HV=Hypervolume_calculation(pf,repoint);
        train_HV=HV(pf,repoint);
        fprintf('Running test function: %s \n %d times train_HV=%f \n', s1,j,train_HV);
        PSdata.MO_Ring_PSO_SCD{j}=ps;
        PFdata.MO_Ring_PSO_SCD{j}=pf;%
        Pfdata.MO_Ring_PSO_SCD{j}=fitness_test;%
        Indicator.MO_Ring_PSO_SCD(j,:)= [test_HV,train_HV, t ] ;
        %% 去掉重复特征获取多个特征子集
        ps=1.*(ps>0.6);
        [ps,ips,~]=unique(ps,'rows');
        pf=pf(ips,:);
        % 找出ps中非零解对应的位置，也就是选择的特征组合
        for i_ps=1:size(ps,1)
            Selected_Fi{i_ps,1}=find(ps(i_ps,:)>0.6);
            Particle{i_ps,1}=[ps(i_ps,:),pf(i_ps,:),[nnz(ps(i_ps,:)) find(ps(i_ps,:)>0.6)]];%把j的特征组合放到i的后边，最前加0，表示间隔
        end
        results{j,1}=Particle;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %% 求解平均精度值
        %% 求解平均精度值
        Obj=pf;
        Dec=ps;
        Front = NDSort(Obj, inf);
        index = find(Front ==1);
        PF1=Obj(index,:); %%%获得第一层前沿的个体
        PS1=Dec(index,:); %%%获取非支配解集
        erro=PF1(:,1);
        index1=find(erro==min(erro));       %%获得最小错误率的最优特征子集
        PS11=PS1(index1,:);
        num1(i_data,j)=nnz(PS11(1,:)>0.6);  %%计算具有最小训练分类错误的的特征子集大小
        %% 评估测试集并计算出测试集的HV指标值
        Nsize=size(PS11(1,:),1); %%具有最小训练分类错误率的特征子集计算其对应的对应的测试分类错误率
        fitness_test=ones(Nsize,2);
        for kk=1:Nsize
            fitness_test(kk,:)=Feature_selection_Fun_deal_conti_x(PS11(kk,:),train_data,test_data);
        end
        Pf1=fitness_test;
        Acc1(i_data,j)=1- (Pf1(:,1));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

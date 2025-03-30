
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
    s1=dataname{i_data,1};    %%Ҫ�洢��������һ������ ��ȡ��һ�����ݼ�
    FileString1= [strcat('train_dataSet_',s1),'.mat']; %%ת���������� ��ȡѵ����
    FileString2= [strcat('test_dataSet_',s1),'.mat']; %%ת����������  ��ȡ���Լ�
    
    load(FileString1,'train_data');
    load(FileString2,'test_data');
    Num_feature=size(train_data,2)-1; %%����������Ҳ�Ǳ��볤�� ��1�������ݱ�ǩ
    %% Initialize parameters in the algorithm
    if Num_feature>300
        popsize=300;
    else
        popsize=Num_feature;
    end
    Max_evaluation=100*popsize;
    Max_Gen=fix(Max_evaluation/popsize);
    fname='FS';  % function name Ŀ�꺯��
    %% ����÷�
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
    repoint=[1,1];          % �ο�������Ϊ[1,1]
    for j=1:runtimes
        %% Search the PSs using NCDE
        %  fprintf('Running test function: %s, times = %d \n', s1,j);
        t1 = cputime;
        [ps,pf]=MBLPE(fname,xl,xu,n_obj,popsize,Max_Gen,train_data);
        t = cputime - t1;
        %% �������Լ�����������Լ���HVָ��ֵ
        Nsize=size(ps,1);
        fitness_test=ones(Nsize,n_obj);
        for kk=1:Nsize
            fitness_test(kk,:)=Feature_selection_Fun_deal_conti_x(ps(kk,:),train_data,test_data);
        end
        %% Indicators-----�������Լ���ָ��ֵ
        %  test_HV=Hypervolume_calculation(fitness_test,repoint);
        test_HV=HV(fitness_test,repoint);
        fprintf('Running test function: %s \n %d times test_HV1=%f \n', s1,j,test_HV);
        %% Indicators-----����ѵ������ָ��ֵ
        %   HV=Hypervolume_calculation(pf,repoint);
        train_HV=HV(pf,repoint);
        fprintf('Running test function: %s \n %d times train_HV=%f \n', s1,j,train_HV);
        PSdata.MO_Ring_PSO_SCD{j}=ps;
        PFdata.MO_Ring_PSO_SCD{j}=pf;%
        Pfdata.MO_Ring_PSO_SCD{j}=fitness_test;%
        Indicator.MO_Ring_PSO_SCD(j,:)= [test_HV,train_HV, t ] ;
        %% ȥ���ظ�������ȡ��������Ӽ�
        ps=1.*(ps>0.6);
        [ps,ips,~]=unique(ps,'rows');
        pf=pf(ips,:);
        % �ҳ�ps�з�����Ӧ��λ�ã�Ҳ����ѡ����������
        for i_ps=1:size(ps,1)
            Selected_Fi{i_ps,1}=find(ps(i_ps,:)>0.6);
            Particle{i_ps,1}=[ps(i_ps,:),pf(i_ps,:),[nnz(ps(i_ps,:)) find(ps(i_ps,:)>0.6)]];%��j��������Ϸŵ�i�ĺ�ߣ���ǰ��0����ʾ���
        end
        results{j,1}=Particle;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  %% ���ƽ������ֵ
        %% ���ƽ������ֵ
        Obj=pf;
        Dec=ps;
        Front = NDSort(Obj, inf);
        index = find(Front ==1);
        PF1=Obj(index,:); %%%��õ�һ��ǰ�صĸ���
        PS1=Dec(index,:); %%%��ȡ��֧��⼯
        erro=PF1(:,1);
        index1=find(erro==min(erro));       %%�����С�����ʵ����������Ӽ�
        PS11=PS1(index1,:);
        num1(i_data,j)=nnz(PS11(1,:)>0.6);  %%���������Сѵ���������ĵ������Ӽ���С
        %% �������Լ�����������Լ���HVָ��ֵ
        Nsize=size(PS11(1,:),1); %%������Сѵ����������ʵ������Ӽ��������Ӧ�Ķ�Ӧ�Ĳ��Է��������
        fitness_test=ones(Nsize,2);
        for kk=1:Nsize
            fitness_test(kk,:)=Feature_selection_Fun_deal_conti_x(PS11(kk,:),train_data,test_data);
        end
        Pf1=fitness_test;
        Acc1(i_data,j)=1- (Pf1(:,1));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end

function f=FS(x,train_data)
% Input
%          x         1*number_of_features                      0 or 1 values 1means the corresponding feature is selected
%       train_data    number_traindata*1+number_of_features    the first column is label
%       test_data     number_testdata*1+number_of_features     the first column is label

% Output
%          f          1*2     the first is number of selected features the second is testing accuracy
f=zeros(1,2);
x1=x;
x1=1.*( x1>=0.6);   %%把连续的x变成离散的
if nnz(x1)==0       %%%nnz（X）矩阵或向量中的非零元素，n=nnz(X)返回矩阵中的非零元素的数目
    f=[1,1];        %%%* 没有选择特征，错误率设置为1
else
    tr=train_data(:,logical([1,x1]));
    k=5;
    %% K折交叉验证
    m1=size(tr,1);
    CV=5;
    indices = crossvalind('Kfold', m1, CV);     %将数据样本随机分割为5部分
    ACC=[];
    for i = 1:CV                                %%循环5次，分别取出第i部分作为测试样本，其余两部分作为训练样本
        test = (indices == i);                  %%一份测试
        train = ~test; %%其他数据训练数据
        trainData = tr(train, :);
        testData = tr(test, :);
        [test_accuracy,~]=f_knn(trainData,testData,k);%%要验证的算法
        ACC(1,i)=test_accuracy;
        erro(1,i)=1-test_accuracy;
    end
    f(1,1)=mean(erro); %%第一个目标
    f(1,2)=nnz(x1)/length(x1); %%特征数量的比值  %%第二个目标
end
end

function [A,knnlabel]=f_knn(tr,te,k)
% tr        训练数据 第一列是类别标签 2：end列是训练样本数据
% te        测试数据 第一列是类别标签 2：end列是测试样本数据
% K         近邻个数

% A         测试精度
% knnlabel  测试样本实际分类标签
data=[tr;te];
n=size(data,2);
label=data(:,1);
L=unique(label);  % 合并A中相同数据
ls=length(L(:));  %统计类别总数
m1=size(tr,1);
m2=size(te,1);

trd=tr(:,2:n);
trl=tr(:,1);
ted=te(:,2:n);
tel=te(:,1);

for j=1:m2
    distance=zeros(m2,1);
    for i=1:m1
        distance(i)=norm(ted(j,:)-trd(i,:)); %%测试数据-训练数据的归一化值
    end
    %选择排序，只找出最小的前K个数据,对数据和标号都进行排序
    [distance1,index]=sort(distance); %以升序排序
    %     distance11=distance1(1:k);
    label=trl(index(1:k));
    
    %出现次数最多的类别标号即为该测试样本的类别标号
    knnlabel(j,1)=mode(label);
end
bj=(knnlabel==tel);
a=nnz(bj);
A=a/m2; %输出识别率
end
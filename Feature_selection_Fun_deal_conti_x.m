function f=Feature_selection_Fun_deal_conti_x(x,train_data,test_data)
% Input
%          x         1*number_of_features                      0 or 1 values 1means the corresponding feature is selected
%       train_data    number_traindata*1+number_of_features    the first column is label
%       test_data     number_testdata*1+number_of_features     the first column is label

% Output
%   f          1*2     the first is number of selected features the second is testing accuracy
f=zeros(1,2);
x1=x;
x1=1.*( x1>=0.6);    %%把连续的x变成离散的
if nnz(x1)==0       %%%nnz（X）矩阵或向量中的非零元素，n=nnz(X)返回矩阵中的非零元素的数目
    f=[1,1];         %%%* 没有选择特征，错误率设置为1
else
    % f(1,1)=nnz(x);   %%返回X中非零元素的个体的数量--其实就是被选择的特征的数量--赋值给第一个目标函数
    te=test_data(:,logical([1,x1])); %%根据当前个体特征被选择的位置，调出测试数据
    tr=train_data(:,logical([1,x1])); %%根据当前个体特征被选择的位置，调出训练数据
    k=5;  %%KNN中的K=5
    [test_accuracy,~]=f_knn(tr,te,k); %%根据当前个体被选择特征选出的训练数据和测试数据来进行KNN 得到测试精度
    f(1,1)=1-test_accuracy;  %识别的错误率
    f(1,2)=nnz(x1)/length(x1); %%特征数量的比值--自己添加
end
end


function [A,knnlabel]=f_knn(tr,te,k)%%训练数据 测试数据 K的大小，测试数据和训练数据中含有数据标签
% tr 训练数据 第一列是类别标签 2：end列是训练样本数据
% te 训练数据 第一列是类别标签 2：end列是测试样本数据
% K 近邻个数

% A        测试精度
% knnlabel 测试样本实际分类标签
data=[tr;te]; %%把选出来的测试数据和训练数据组合在一起
n=size(data,2);
label=data(:,1);  %%得出标签列
L=unique(label);  %%合并label中相同数据 ---分类几类
ls=length(L(:));  %统计类别总数---计算有几个标签--几个类别
m1=size(tr,1);    %%得出训练数据的数量
m2=size(te,1);    %%得出测试数据的数量

trd=tr(:,2:n);   %%拿出训练数据中除标签外的其他数据
trl=tr(:,1);     %%拿出训练数据中标签数据
ted=te(:,2:n);   %%拿出测试数据中除标签外的其他数据
tel=te(:,1);     %%拿出测试数据中标签数据

for j=1:m2 %%测试数据的大小
    distance=zeros(m2,1);
    for i=1:m1
                 distance(i)=norm(ted(j,:)-trd(i,:)); %%两个差值的范数--测试数据中每个数据与训练数据集之间欧式距离
%         distance(i)=sqrt(sum((ted(j,:)-trd(i,:)).^2));
    end
    %% 选择排序，只找出最小的前K个数据,对数据和标号都进行排序--这里的K=1
    [distance1,index]=sort(distance); %以升序排序
    % distance11=distance1(1:k);
    label=trl(index(1:k)); %%找出距离最近的数据的标签
    
    %% 出现次数最多的类别标号即为该测试样本的类别标号
    knnlabel(j,1)=mode(label); %%mode(X)计算向量或数组X中出现次数最多的数值；对于矩阵则返回每列元素中频率最多的元素
end

bj=(knnlabel==tel); %%%判断通过距离得出的标签是否是测试数据的标签一致，一致为1，否则为0
a=nnz(bj); %%%返回非零元素个数的数量
A=a/m2; %%%输出识别率 非零元素的数量/总数量  预测一致的数量与测试数据总是的标记 ---识别的精度
end
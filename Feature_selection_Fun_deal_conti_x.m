function f=Feature_selection_Fun_deal_conti_x(x,train_data,test_data)
% Input
%          x         1*number_of_features                      0 or 1 values 1means the corresponding feature is selected
%       train_data    number_traindata*1+number_of_features    the first column is label
%       test_data     number_testdata*1+number_of_features     the first column is label

% Output
%   f          1*2     the first is number of selected features the second is testing accuracy
f=zeros(1,2);
x1=x;
x1=1.*( x1>=0.6);    %%��������x�����ɢ��
if nnz(x1)==0       %%%nnz��X������������еķ���Ԫ�أ�n=nnz(X)���ؾ����еķ���Ԫ�ص���Ŀ
    f=[1,1];         %%%* û��ѡ������������������Ϊ1
else
    % f(1,1)=nnz(x);   %%����X�з���Ԫ�صĸ��������--��ʵ���Ǳ�ѡ�������������--��ֵ����һ��Ŀ�꺯��
    te=test_data(:,logical([1,x1])); %%���ݵ�ǰ����������ѡ���λ�ã�������������
    tr=train_data(:,logical([1,x1])); %%���ݵ�ǰ����������ѡ���λ�ã�����ѵ������
    k=5;  %%KNN�е�K=5
    [test_accuracy,~]=f_knn(tr,te,k); %%���ݵ�ǰ���屻ѡ������ѡ����ѵ�����ݺͲ�������������KNN �õ����Ծ���
    f(1,1)=1-test_accuracy;  %ʶ��Ĵ�����
    f(1,2)=nnz(x1)/length(x1); %%���������ı�ֵ--�Լ����
end
end


function [A,knnlabel]=f_knn(tr,te,k)%%ѵ������ �������� K�Ĵ�С���������ݺ�ѵ�������к������ݱ�ǩ
% tr ѵ������ ��һ��������ǩ 2��end����ѵ����������
% te ѵ������ ��һ��������ǩ 2��end���ǲ�����������
% K ���ڸ���

% A        ���Ծ���
% knnlabel ��������ʵ�ʷ����ǩ
data=[tr;te]; %%��ѡ�����Ĳ������ݺ�ѵ�����������һ��
n=size(data,2);
label=data(:,1);  %%�ó���ǩ��
L=unique(label);  %%�ϲ�label����ͬ���� ---���༸��
ls=length(L(:));  %ͳ���������---�����м�����ǩ--�������
m1=size(tr,1);    %%�ó�ѵ�����ݵ�����
m2=size(te,1);    %%�ó��������ݵ�����

trd=tr(:,2:n);   %%�ó�ѵ�������г���ǩ�����������
trl=tr(:,1);     %%�ó�ѵ�������б�ǩ����
ted=te(:,2:n);   %%�ó����������г���ǩ�����������
tel=te(:,1);     %%�ó����������б�ǩ����

for j=1:m2 %%�������ݵĴ�С
    distance=zeros(m2,1);
    for i=1:m1
                 distance(i)=norm(ted(j,:)-trd(i,:)); %%������ֵ�ķ���--����������ÿ��������ѵ�����ݼ�֮��ŷʽ����
%         distance(i)=sqrt(sum((ted(j,:)-trd(i,:)).^2));
    end
    %% ѡ������ֻ�ҳ���С��ǰK������,�����ݺͱ�Ŷ���������--�����K=1
    [distance1,index]=sort(distance); %����������
    % distance11=distance1(1:k);
    label=trl(index(1:k)); %%�ҳ�������������ݵı�ǩ
    
    %% ���ִ�����������ż�Ϊ�ò��������������
    knnlabel(j,1)=mode(label); %%mode(X)��������������X�г��ִ���������ֵ�����ھ����򷵻�ÿ��Ԫ����Ƶ������Ԫ��
end

bj=(knnlabel==tel); %%%�ж�ͨ������ó��ı�ǩ�Ƿ��ǲ������ݵı�ǩһ�£�һ��Ϊ1������Ϊ0
a=nnz(bj); %%%���ط���Ԫ�ظ���������
A=a/m2; %%%���ʶ���� ����Ԫ�ص�����/������  Ԥ��һ�µ�����������������ǵı�� ---ʶ��ľ���
end
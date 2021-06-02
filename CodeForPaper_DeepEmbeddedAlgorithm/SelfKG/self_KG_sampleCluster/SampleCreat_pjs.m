clear ; close all; clc;warning('off');
% sample_statlog=xlsread('帕金森总数据集_total2.xlsx','sacker','A2:AB1209');
sample_statlog=xlsread('KG.xlsx','sheet1','A2:AB1171');
M = 200;%设置偏差与数据的比值
[m_s,n_s]=size(sample_statlog);
Y=sample_statlog(:,end);
type_num = size(unique(Y),1);
class=cell(1,type_num);
chouqu_ratio=0.8;%%抽取样本率，从训练集中抽取
    if min(unique(Y))==0
        sample_statlog(:,end)=sample_statlog(:,end)+1; %确保标签是从0开始，但是还需要人工确保其是连续自然数
    end
for i=1:m_s
    for j=1:type_num
        if sample_statlog(i,end)==j
            class{1,j}=[class{1,j};sample_statlog(i,:)];
        end
    end
end
%% 分训练、验证和测试样本
train_data=[];
valid_data=[];
test_data=[];
% ID_label_i=size(unique(class{1,i}(:,1)),1);
for i=1:type_num
    m_class=size(unique(class{1,i}(:,1)),1);
    class{1,i}(:,1)=class{1,i}(:,1)-min(class{1,i}(:,1))+1;
    n=round(m_class/3);
    class_rand=randperm(m_class);
        for j=1:size(class_rand,2)
            if j<n+1
                data_in=find(class{1,i}(:,1)==class_rand(1,j));
                train_data=[train_data;class{1,i}(data_in,:)];
            elseif j<2*n+1
                data_in=find(class{1,i}(:,1)==class_rand(1,j));
                valid_data=[valid_data;class{1,i}(data_in,:)];
            else
                data_in=find(class{1,i}(:,1)==class_rand(1,j));
                test_data=[test_data;class{1,i}(data_in,:)];
            end
        end
end
train_data_rand=train_data(randperm(size(train_data,1)),:);
valid_data_rand=valid_data(randperm(size(valid_data,1)),:);
test_data_rand=test_data(randperm(size(test_data,1)),:);

trainX=train_data_rand(:,2:end-1);
trainY=train_data_rand(:,end);
validX=valid_data_rand(:,2:end-1);
validY=valid_data_rand(:,end);
testX =test_data_rand(:,2:end-1);
testY =test_data_rand(:,end);

% mean_X=mean(trainX);
% U_train=1/M*repmat(mean_X,size(trainX,1),1).*(rand(size(trainX,1),size(trainX,2))*2-1);
% U_valid=1/M*repmat(mean_X,size(validX,1),1).*(rand(size(validX,1),size(validX,2))*2-1);
% U_test =1/M*repmat(mean_X,size(testX,1),1).*(rand(size(testX,1),size(testX,2))*2-1);
% U1{1,1}=U_train; U1{1,2}=U_valid; U1{1,3}=U_test;%记录所有的样本的偏差
% 
% for i=1:3 %% 抽取0.8的训练集寻找W
% train=[trainX,trainY];
% in=randperm(size(train,1));
% U4=U1{1,1}(in(1,1:chouqu_ratio*size(train,1)),:);%记录所有抽取的样本的偏差，与样本对应
% train_rand=train(in(1,1:chouqu_ratio*size(train,1)),:);
% train_X{1,i}=train_rand;%记录抽样样本
% Uc{1,i}=U4;%记录所有的抽样样本的抽样偏差
% end
% save('sample_statlog.mat','trainX','trainY','validX','validY','testX','testY','type_num');
% save('Uc_satalog.mat','Uc');%%为了实验过程中抽取样本的噪声一致，单独保存抽取样本噪声文件
% save('U1_satalog.mat','U1');%%为了实验过程中加入噪声一致，单独保存噪声文件
% save('train_X.mat','train_X');%%加载实验抽取训练数据集
save('pd_self_KG.mat','trainX','trainY','validX','validY','testX','testY','type_num');
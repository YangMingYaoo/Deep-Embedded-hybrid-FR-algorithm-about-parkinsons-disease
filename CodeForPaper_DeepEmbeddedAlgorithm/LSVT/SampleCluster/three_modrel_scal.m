% 此为深度样本空间构造程序，在运行该深度聚类的代码之前先把原始数据集分成训练、验证和测试，之后仅仅堆训练集进行深度样本聚类，不改变验证集和测试集。
%之后单独对训练集进行2次深度空间聚类生成2个深度样本空间，都是用k-means进行构造
clc;close all;clear all;
tic;load 'ttraintestdata';
%% 测试部分 
%加载数据集

% TEST = [testX testY];
% TRAIN = [trainX trainY];
% DATA  =[TRAIN;TEST];
% % xlswrite('YYY.xlsx',); % X为要保存的矩阵变量，YYY为保存后的名称
%  xlswrite('LSVT_DATA.xlsx',DATA); 
%% 


% 由于原来的mat数据是分类的，为了分类别来聚类数据，所以准备把数据合起来，分成不同的类别
train_data = [trainX trainY]; %把训练数据和标签放到一起
test_data = [testX testY];   %把测试数据和标签放到一起
trainX = train_data (1:42,:);
valid_data = train_data (43:84,:);
%% 按照类别拆分训练数据 程序段
%获取训练数据矩阵的行数和列数
[m_s,n_s] = size(trainX); 
Y = trainX(:,end); % [取出训练数据最后一行，即标签]单独的一个end就是矩阵的最后一列
type_num = size(unique(Y),1);  % 计算标签的类别数目，这里返回为2，为2类数据
%分开类别程序段
class = cell(1,type_num);     % 使用cell()函数建立一个1行2列的元胞数组。
for i = 1:m_s                                    % m_s为数据样本行数，for循环作用，经过循环后把两类样本分开存放在上面建立好的元胞数组里面，在元胞数组中刚好有2类，故正好存放2类样本。
    for j = 1:type_num                           % type_num 为类别数目。
        if trainX(i,end) == j
            class{1,j}= [class{1,j};trainX(i,:)];
        end
    end
end
%% 对每一类样本进行深度空间聚类
iter = 3;
traindataX = cell(1,iter);
for i = 1 : type_num
    %每一层样本输出率
    % size()
    P = 0.8; % 每一层样本输出率
    %对不是整数的数值四舍五入变成整数
    data1 = class{1,i};
    [m_s1,n_s1] = size(data1); 
    %第一层聚类
    N1 =  round( m_s1 * P);
    data2 = sampleCluster(data1,N1);
    %第二层聚类 
    N2 =  round(N1 * P);
    data3 = sampleCluster(data1,N2);
    traindataX{1,1} = [traindataX{1,1};data1]
    traindataX{1,2} = [traindataX{1,2};data2];
    traindataX{1,3} = [traindataX{1,3};data3]
end
%traindataX是原始空间数据集经过2次K均值聚类之后组成的深度样本空间。不改变验证集和测试集

%for循环加上把深度样本空间中每一个子空间的中的训练数据打乱
for j = 1:iter
    traindataX{1,j} = traindataX{1,j}(randperm(size( traindataX{1,j},1)),:);
end

%之后把训练、验证和测试集合组成的新的数据集保存下来，供后面使用。
%保存数据,保存之后的数据都是已打乱的数据
save('PD_LSVTsample1.mat','traindataX','valid_data','test_data','type_num');
toc

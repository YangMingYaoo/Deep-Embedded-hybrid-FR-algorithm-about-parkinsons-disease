function [train_data,valid_data,test_data] = MaxLittle_PD_sample_creat(MaxLittle_OriginalData195X24)
[m_s,n_s]=size(MaxLittle_OriginalData195X24); %获取原始数据矩阵的行数和列数
data = MaxLittle_OriginalData195X24;  
Y = data(:,end); % [取出拼接好矩阵的最后一行，即标签]单独的一个end就是矩阵的最后一列
type_num = size(unique(Y),1);  % 计算标签的类别数目，这里返回为2，为2类数据
class=cell(1,type_num);     % 使用cell()函数建立一个1行2列的元胞数组。
%% 在sakar数据集中，标签为0的是有病的，为了符合巴铁留学生的算法的思路，先把为0的标签设置为2. 
   %（其实设置为2是为了匹配LSVT数据集，因为在LSVT数据集中，语音不能接受的，即还是认为有病人的语音的样本设置标签为2）
%修改数据标签符合程序的设置，PWP设置为2，Helthy设置为1
   for i = 1:m_s 
    if data(i,end) == 0
       data(i,end) = data(i,end) + 1;
    elseif data(i,end) == 1
           data(i,end) = data(i,end) + 1;
    end 
end                               % 
%% for循环说明  
% 下面这个for循环把样本按照标签分开，样本标签为1和2，2为有病，1为健康人，
% 所以2个class中，第一个class中存放的是第1类标签的人，即第1类中存放的是健康人，第2类中存放的是右PD患者的。
for i = 1:m_s                                    % m_s为数据样本行数，for循环作用，经过循环后把两类样本分开存放在上面建立好的元胞数组里面，在元胞数组中刚好有2类，故正好存放2类样本。
    for j = 1:type_num                           % type_num 为类别数目。
        if data(i,end) == j  
            class{1,j} = [class{1,j};data(i,:)];
        end
    end
end     

%% 
% 对于 MaxLittle数据集， 一共两类。PWP和健康人，在每一类中按照人来分，训练集0.6个比列 验证集取0.2个比例，测试集取0.2个比例，
% 一个人的所有样本，保证在一个集中，要么在训练集，要么在测试集，要么在验证集中
% 比如在maxlittle数据集中，一个人有6个样本，这6个样本要么在训练集，要么在测试集，要么在验证集中        

train_data1=[];   % 建立空数据组，分数据之前先建好空的矩阵，有点先占坑的感觉。占好坑之后再往里面加东西 
valid_data1=[];
test_data1=[];
train_data2=[];   % 建立空数据组，分数据之前先建好空的矩阵，有点先占坑的感觉。占好坑之后再往里面加东西 
valid_data2=[];
test_data2=[];
indexcase1 = [];
indexcase2 = [];
for i =  1:type_num
    m_class = size(class{1,i},1); %选中class 中第一行，第i列。
    class_rand = class{1,i};
    index1 = [];
    index0 = [];
     switch i
         case  2  % 有病的人
         % 训练集
         for j = 9 : 16           %12代表我要在在case2中取12个人，其余的方位验证集核测试集
            index0 = [index0;find(class_rand(:,1) == j)];  %训练集
         end
         train_data2 = [train_data2; class_rand(index0,:)];
         
         % 验证集
         for j = 17 : 24
              index1 =  [index1;find(class_rand(:,1) == j)];
         end
         valid_data2 = [valid_data2;class_rand(index1,:)];
         
         % 测试集
         indexcase2 = [indexcase2;index0;index1]
         class_rand(indexcase2,:) = [];
         test_data2 = class_rand;

        case 1  %健康人
%         for k = 10 : (10 - 1) + round(length(unique(class_rand(:,1))) / 3) 
%         index0 = find(class_rand(:,1) == (2 * k + 1))
%         index1 = find(class_rand(:,1) == (2 * k + 2))
%         train_data1 = [train_data1; class_rand(index0,:)];
%         valid_data1 = [valid_data1 ;class_rand(index1,:)];
%         indexcase1 = [indexcase1 ;index0;index1]
%         end
%         class_rand(indexcase1,:) = [];
%         test_data1 = class_rand; 
%          train_data1 = class_rand(1:312,:);%12人，每人26个样本，一共312个样本
%          valid_data1 = class_rand(313:416,:);
%          test_data1 = class_rand(417:520,:)
%      end
  % 训练集
         for j = 1: 3           %12代表我要在在case2中取12个人，其余的方位验证集核测试集
            index0 = [index0;find(class_rand(:,1) == j)];  %训练集
         end
         train_data1 = [train_data1; class_rand(index0,:)];
         
         % 验证集
         for j = 4 : 6
              index1 =  [index1;find(class_rand(:,1) == j)];
         end
         valid_data1 = [valid_data1;class_rand(index1,:)];
         
         % 测试集
         indexcase1 = [indexcase1;index0;index1]
         class_rand(indexcase1,:) = [];
         test_data1 = class_rand;
     end

       
%% 上面程序段划分好之后，标签排列仍然比较正切，所以需要把按照打乱标签的行，从而把样本的行也打乱。

% train_data_rand = train_data(randperm(size(train_data,1)),:); %再次打断训练集
% valid_data_rand = valid_data(randperm(size(valid_data,1)),:);%再次打断验证集
% test_data_rand = test_data(randperm(size(test_data,1)),:);%再次打断测试集
% 因为上面for循环把数据集取出来是不是随机取出来的，而是按照样本的顺序截取的，所以紧挨着上面3行是把数据集打断，有利于训练。

% trainX = train_data_rand; 
% validX = valid_data_rand; 
% testX =test_data_rand;
end    
train_data = [train_data1;train_data2]   % 总的训练集
valid_data = [valid_data1;valid_data2]   % 总的验证集
test_data = [test_data1; test_data2]     % 总的测试集
   
end
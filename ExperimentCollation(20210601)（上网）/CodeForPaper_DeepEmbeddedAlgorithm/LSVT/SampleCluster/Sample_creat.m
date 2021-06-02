
function[trainX,trainY,validX,validY,testX,testY,type_num]=Sample_creat(trainX,trainY,validX,validY,testX,testY)
sample_statlog=[[trainX trainY];[validX validY];[testX testY]]; %把训练 验证 测试集 数据以及标签组合到一起[把原始加载出来的切分好的数据集 使用矩阵【上下拼接】的方式拼接好]
[m_s,n_s]=size(sample_statlog); %获取拼接好之后数据矩阵的行数和列数
Y=sample_statlog(:,end); % [取出拼接好矩阵的最后一行，即标签]单独的一个end就是矩阵的最后一列
type_num = size(unique(Y),1);  % 计算标签的类别数目，这里返回为2，为2类数据
class=cell(1,type_num);     % 使用cell()函数建立一个1行2列的元胞数组。
    if min(unique(Y))==0    %如果最小的列标签为0 ，加1 
        sample_statlog(:,end)=sample_statlog(:,end)+1;
    end
for i=1:m_s                                    % m_s为数据样本行数，for循环作用，经过循环后把两类样本分开存放在上面建立好的元胞数组里面，在元胞数组中刚好有2类，故正好存放2类样本。
    for j=1:type_num                           % type_num 为类别数目。
        if sample_statlog(i,end)==j
            class{1,j}=[class{1,j};sample_statlog(i,:)];
        end
    end
end     % for循环结束后刚好把组装好的2类样本，拆分开放到元胞数组中，从分开的数据显示，1类有42个样本，2类有84个样本，明显处于不平衡的样本数据集。


%% 分训练、验证和测试样本
train_data=[];   % 建立空数据组，分数据之前先建好空的矩阵，有点先占坑的感觉。占好坑之后再往里面加东西 
valid_data=[];
test_data=[];
for i=1:type_num
    m_class=size(class{1,i},1);
    n=round(m_class/3);     %四舍五入函数，返回一个最接 近的整数
    class_rand=class{1,i}(randperm(m_class),:);  %randperm(m_class) 随机打乱目标数，用在这里的目的是随机的取出来每1类的样本
    train_data=[train_data;class_rand(1:n,:)];  %把随机取出来的每1类样本取1/n添加进 train_data中
    valid_data=[valid_data;class_rand(1+n:2*n,:)]; % 取1/n放进valid_data
    test_data=[test_data;class_rand(1+2*n:end,:)];%取1/n放进test_data
end
 %上面这个for循环作用是把 每一类样本都分成3份，也就是把每一类样本都分成训练集，验证集以及测试集合。

train_data_rand = train_data(randperm(size(train_data,1)),:); %再次打断训练集
valid_data_rand = valid_data(randperm(size(valid_data,1)),:);%再次打断验证集
test_data_rand = test_data(randperm(size(test_data,1)),:);%再次打断测试集
% 因为上面for循环把数据集取出来是不是随机取出来的，而是按照样本的顺序截取的，所以紧挨着上面3行是把数据集打断，有利于训练。

trainX = train_data_rand(:,1:end-1);  %end代表最后一列标签列，end-1代表往前移动一列，在这里就是只取出来训练数据集，不取标签
trainY = train_data_rand(:,end);
validX = valid_data_rand(:,1:end-1); %下面这几个一样的作用
validY = valid_data_rand(:,end);
testX =test_data_rand(:,1:end-1);
testY =test_data_rand(:,end);
% 把上面分好的数据存储到sample_PD.mat,方便后面调用
save('sample_PD.mat','trainX','trainY','validX','validY','testX','testY','type_num');
end


%读完个sample文件我们能理解这个函数的作用了吗？
% 先把原始的分好的数据通过矩阵拼接 组装到一起，在按类别分开，在每个类别中都把数据分成3份类型数据，分别为训练 测试 和验证，每一类数据42个样本...
%2个类别的中同一类型的数据放在一起，在打乱， 之后再从打乱中的数据中把特征和标签分类，最终存储这个最终的数据
%那么这样做的目的是什么 ？ 
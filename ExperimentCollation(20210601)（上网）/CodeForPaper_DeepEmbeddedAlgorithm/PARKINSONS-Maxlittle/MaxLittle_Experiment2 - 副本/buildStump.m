%传进来每一折交叉验证的训练数据和训练数据标签，以及样本的权重。
%把  弱分类找出来 然后在返回去。
% 该函数的作用就是找最基本的分类器，从最基本的分类器中找出错误率最小的，返回上一个函数。
function stump = buildStump(X, y, weight)   
D = size(X, 2); % 得到数据集的维度，也就是特征数。

if nargin <= 2
    weight = ones(size(X,1), 1);
end

cellDS = cell(D, 1);
Err = zeros(D, 1);
for i = 1:D
    cellDS{i} = buildOneDStump(X(:,i), y, i, weight);   %调用函数
    Err(i) = cellDS{i}.error;
end

%cellDS 代表什么 
[v, idx] = min(Err);
stump = cellDS{idx}; %在这里找出最小错误率对应的stump返回到上一个函数。
end
%该函数输出刚开始的弱分类器。强弱分类器的定义在Adaboost中有定义。 
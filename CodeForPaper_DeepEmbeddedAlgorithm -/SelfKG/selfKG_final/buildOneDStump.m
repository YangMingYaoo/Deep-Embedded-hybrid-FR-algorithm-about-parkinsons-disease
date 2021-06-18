function stump = buildOneDStump(x, y, d, w)
%先来分析函数传输进入的什么。传进去，数据的所有行中的一列，也就是取出所有数据集的某一列输入进去。
%样本的标签，以及决定哪一维数据的d,以及样本的权重

[err_1, t_1] = searchThreshold(x, y, w, '>'); % > t_1 -> +1  %调用searchThreshold函数,该函数就在本函数的下面
[err_2, t_2] = searchThreshold(x, y, w, '<'); % < t_2 -> +1
%要知道searchThreshold函数返回的是:err_1,和t_1
% 调用initStump()函数                 %传进来x的第一列,(每次传进来一类,310维度所以传进来310次,)                   
stump = initStump(d);

if err_1 <= err_2
    stump.threshold = t_1;
    stump.error = err_1;
    stump.less = -1;
    stump.more = 1;%
else
    
    stump.threshold = t_2;
    stump.error = err_2;
    stump.less = 1;
    stump.more = -1;
end
end
%这个函数是干啥的,搜索出门槛? 
function [error, thresh] = searchThreshold(x, y, w, sign)
N = length(x);
err_n = zeros(N, 1);
y_predict = zeros(N, 1);
for n=1:N
    switch sign           
        case '>'                          % 由于logical（）函数是逻辑函数，所以
            idx = logical(x >= x(n));    %这里的X(n)相当于一个门槛,X(n)随着n取值的不同表示的数字也不一样。
            y_predict(idx) = 1;
            y_predict(~idx) = -1;
        case '<'
            idx = logical(x < x(n));
            y_predict(idx) = 1;
            y_predict(~idx) = -1;
    end
    err_label = logical(y ~= y_predict);
    err_n(n) = sum(err_label.*w)/sum(w); %这里为什么要/sum(w)  ，%因为是在一个循环中，放在循环中之后，必须和原理上有所区别
end                                      %这一步用来计算每个样本在分类器上的误差率。
[v, idx] = min(err_n);
error = v;
thresh = x(idx);
end
%函数调用2次searchThreshold,一次符号是>,另外一次符号是<.在大于中来一遍,在小于中来一遍,算出是大于号的误差小,还是小于号的误差小.找出最小错误率对应的门槛

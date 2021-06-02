function abClassifier = buildAdaBoost(trnX, trnY, iter, tstX, tstY) %传进来每一折交叉验证的数据 训练和测试 
if nargin < 4
    tstX = []; 
    tstY = [];
end                                 
abClassifier = initAdaBoost(iter); %调用初始化Adaboost函数。
% 下面开始建立Adaboost,这个代码基本和Adaboost的原理是一样的.
N = size(trnX, 1); % Number of training samples,训练样本的数量
sampleWeight = repmat(1/N, N, 1);   %初始化样本权重

for i = 1 : iter
    weakClassifier = buildStump(trnX, trnY, sampleWeight);  %调用该函数 找到最基本的弱分类器，并从这些基本的弱分类器中找出错误率最小的返回,利用最小误差的分类器进行下一步。
    abClassifier.WeakClas{i} = weakClassifier;%这里存储每一轮迭代的弱分类器
    abClassifier.nWC = i;
    % Compute the weight of this classifier  计算权重
    %计算返回来的弱分类器系数，可以理解为分类器的权重
    abClassifier.Weight(i) = 0.5*(log((1-weakClassifier.error)/weakClassifier.error));  % 更新弱分类器权重
    % Update sample weight 更新全新过程
    label = predStump(trnX, weakClassifier);                                               %下面开始更新样本的权重，从一开始的每个样本的权重一样，到现在的更新 后就变得不一样了
    tmpSampleWeight = -1*abClassifier.Weight(i)*(trnY.*label); % N x 1
    tmpSampleWeight = sampleWeight.*exp(tmpSampleWeight); % N x 1
    sampleWeight = tmpSampleWeight./sum(tmpSampleWeight); % Normalized标准化    
    % 上面这几句是更新样本的权重。按照Adaboost原理写的，没毛病
    
    
    % Predict on training data
    % 调用predAdaBoost()函数，
    [ttt, abClassifier.trnErr(i)] = predAdaBoost(abClassifier, trnX, trnY);
    % Predict on test data
    if ~isempty(tstY)
        abClassifier.hasTestData = true;
        [ttt, abClassifier.tstErr(i)] = predAdaBoost(abClassifier, tstX, tstY);
    end
    % fprintf('\tIteration %d, Training error %f\n', i, abClassifier.trnErr(i));
end
end

% 对于该函数要清楚，要找的只有一个就是权重。
% 关心权重怎么来的。 是谁的权重，权重是怎么返回主函数的。
function [alpha] = FInd_Alpha(Trainingdata)
% function [alpha] = FInd_Alpha()
% load('Trainingdata.mat')
% [trainX, mu, sigma] = featureCentralize(trainX);%%将样本标准化（服从N(0,1)分布）
% testX = bsxfun(@minus, testX, mu);
% testX = bsxfun(@rdivide, testX, sigma);
% trainX1=trainX(1:42,:);
% trainY1=trainY(1:42,:);
% trainX2=trainX(43:84,:);
% trainY2=trainY(43:84,:);
alpha=[];
% for o=1:2;
    for p = 1:size(Trainingdata)
        X = Trainingdata{p,1};
        trainX = X(:,1:end-1);     %trainX和trainY 是从Trainingdata 取出来的数据
        trainY = X(:,end);
        [ind] = find(trainY==2);
        trainY(ind,1) = -1;         
        nfold = 10;
        iter = 3;
        tstError = zeros(nfold, iter);%测试数据集误差   10行三列，用来存放
        trnError = zeros(nfold, iter);%训练数据集误差
        %调用第一个函数，输出什么？该函数用来建立交叉矩阵，输出测试集的交叉矩阵和训练集的交叉矩阵。
        [trnM, tstM] = buildCVMatrix(size(trainX, 1), nfold); %建立交叉矩阵,看一些交叉矩阵是怎样建立的
        %开始 nfold折交叉验证. 
        for n = 1:nfold          
            fprintf('\tFold %d\n', n);          
            %logical()函数，将数值转化为逻辑值，非零元素转化为1,0元素转化为0.
            idx_trn = logical(trnM(:, n) == 1); %感觉这一步有点多余，因为trnM本来就是就是1和0.
            trnX = trainX(idx_trn, :); %取出每一折交叉验证的训练数据X
            tstX = trainX(~idx_trn, :); %取出每一折交叉验证的测试数据X
            trnY = trainY(idx_trn);    %取出每一折交叉验证的训练数据标签Y
            tstY = trainY(~idx_trn);  %取出每一折交叉验证的测试数据标签Y          %走到这里发现tstM 没用上,一个trnM就可以搞定了 
            %调用第二个函数，输入每一折交叉验证的训练数据、训练数据标签、测试数据、测试数据标签，那么输出什么呢?
            abClassifier = buildAdaBoost(trnX, trnY, iter, tstX, tstY);
            %10折交叉验证数据集，每一折数据都输入进入一次，每一折数据输入后跌带3次，得到3个权重。
            %10折输入进去最终还是3个权重，因为每一折都在更新的还是这三个权重。 
            trnError(n, :) = abClassifier.trnErr;
            tstError(n, :) = abClassifier.tstErr;
        end
        A = abClassifier.Weight; %三次迭代后的 分类器权重
        B = sum(A)/3;            % 对三次迭代后的分离器权重进行求平均値。
        alpha(p)=B;            %把三个子空间数据集权重的平均值返回给主函数。 

% plot(1:iter, mean(trnError, 1)); % Training error
% hold on;
% plot(1:iter, mean(tstError, 1));% Test error
%     end
    end
end


clc;clear ;warning('off');
load('ttraintestdata.mat');%数据集读取
% for i=1:3
% [Data]=Mixdata(Data,Labels);
% data=Data(:,1:end-1);
% labels=Data(:,end);
% trainC=data(1:84,:);
% testX=data(85:end,:);
% testY=labels(85:end,1);

ACC_w1_all=[];
tr2=[];
for J=1:1  %此for循环代表整个框架运行10次，最终的精度取10次运行的平均值 
    FinalAcc=[];
    ensemble=[];
    for s=1:3           % S只是一共有三个子空间的数据样本
%         load('traintestdata.mat');%数据集读取 ,这里为什么要记加载两个数据集？
        trainX1 = train_x{J,s}(:,1:end-1);
        trainY1 = train_x{J,s}(:,end);
%       trainX1 = trainX(1:42,:);
%       trainY1 = trainY(1:42,1);
        validX = trainX(43:84,:);  %这里为何要从第43行开始采集呢？ 因为要从trainX中分离验证集
        validY = trainY(43:84,1);
       % 样本标准化部分 ,suo you de shuju dou tongyi chuli 
       [trainX1, mu, sigma] = featureCentralize(trainX1);%%将样本标准化（服从N(0,1)分布）
       testX = bsxfun(@minus, testX, mu);
       testX = bsxfun(@rdivide, testX, sigma);
       validX = bsxfun(@minus, validX, mu);
       validX = bsxfun(@rdivide, validX, sigma);    
       % trainX1=trainX(1:42,:);
       % trainY1=trainY(1:42,1);
       % validX=trainX(43:84,:);
       % validY=trainY(43:84,1);
       % [trainX1,trainY1,validX,validY,testX,testY,type_num]=Sample_creat(trainX1,trainY1,validX,validY,testX,testY)
       a=[];
       m = 1;
       p=0;
       % l=1;
       tic
       iter=3; % number of iterations   ，iter指对每一个数据集子空间 迭代训练多少次
       Models = cell(iter,1); % For saving the models from each iteration 
       Us = cell(iter,1); % For saving the Us from each iteration
       Trainingdata = cell(iter,1); % For saving the training data from each iteration to calculate the corresponding Alpha
       % while (m>0.0001)
       %% 
       for z=1:iter
           % [fea,U,model,indx,misX,misY] = adbstLDPP(trainX,trainY,testX,testY,type_num);
           
           [fea,U,model,indx,misX,misY] = adbstLDPP(trainX1,trainY1,validX,validY,type_num);
           %? 上一句代码运行时找出错分样本，找出错分样本之后并没有对错分的样本采取什么措施啊？？
           Us{z}=U;
           feature{z,1}=fea; % Saving the features for each iteration
           Trainingdata{z,1} = [trainX1 trainY1]; % combining training data with its labels. Since in each iteration training data will be changed, we need to save corresponding labels
           Medels{z,1}=model;
           trainX1 = [misX;trainX1]; % combining the training data with miss classified samples
           trainY1 = [misY;trainY1]; % combining the training lebels with miss classified labels
           % 这一步对应的是论文流程图里面说的 找En吗?
           for i = 1:size (misY)       % iter=3 .时候分错样本为0个，就没法在计算相关系数矩阵了。
               b = corr(misX(:,i),misY); % Finding the error. however result id NaN because miss classified samples belong to single class.
               a = [a b];
           end                
               
       end                  %我怎么觉的 这一段代码有问题，应该把53和54行放在 51行之前，
       % 到这里后已经把三次迭代循环得到的分类器训练完成了,下面开始找三次训练好的分类器的权重（也可以是多次，在这里是三次。）
       %% 
       %我们应该了解一下Trainingdata 这个数据是什么 ？
       % Finding Alpha  
       alpha = FInd_Alpha(Trainingdata);   % 这里原来解释不对，这很应该是返回的分类器权重。% finding the alpha for training data of each iteration
       % alpha(1)=0.5;

       % Test process 测试处理过程
       svml8 = [];
       svm_prede=[]

       for i = 1:iter  %对每一个子空间的数据做如下处理
           %这里把之前训练三个分类器时保存的东西（LDPP映射矩阵US，训练的分类模型model,特征选择出来的特征relief）再次提取出来
           test1 = testX * (Us{i}); % Coressponding U
           mod1 = Medels{i};  % Corresponding model
           fea = feature{i};  % Coressponding Feature
           test1 = test1(:,fea);
           svm_pred1 = svmpredict(testY,test1,mod1);
           [ind] = find(svm_pred1==2); % Because Sgn function works with labels -1 and 1 so here I changed all my labels from 2 to -1 to make it possible(1,-1) before saving
           svm_pred1(ind,1)= -1;
           svm_pred1 = svm_pred1 * alpha(i);  %alpha（i）代表之前返回的每一个数据子空间样本分类器的权重 。  % Multiplication of alpha with corresponding prediction ，
           svm_prede = [svm_prede svm_pred1];
       end

       % for i=1:iter;
       % test1=testX*(Us{i}); % Coressponding U
       % mod1=Medels{i}% Corresponding model
       % fea=feature{i};% Coressponding Feature
       % test1=test1(:,fea(1:100));
       % svm_pred1 = svmpredict(testY,test1,mod1);
       % [ind]=find(svm_pred1==2); % Because Sgn function works with labels -1 and 1 so here I changed all my labels from 2 to -1 to make it possible(1,-1) before saving
       % svm_pred1(ind,1)=-1;
       % svm_pred1=svm_pred1*alpha(i); % Multiplication of alpha with corresponding prediction
       % svm_prede=[svm_prede svm_pred1];
       % end

      %% Final Prediction 最终的预测 
       [ind]= find(testY==2); % Changing labels having value 2 to -1 for calculation of accuracy    ，因为之前把预测出来的标签修改成了1和-1 ，后面要和testY进行对比，所以也要把testy编程1和-1
       testY(ind,1) = -1;
       Final_predict = sum(svm_prede,2);
       Result = sign(Final_predict);    %sign函数把大于0的设置为1，小于0 的设置为-2，等于0的设置为0。
       Final_Accuracy_with_Adaboost = mean(Result == testY) * 100 % Final accuracy 。因为之前把testY变成了-1和1，所以可以和Result进行对比
       ensemble = [ensemble Result];
       FinalAcc = [FinalAcc Final_Accuracy_with_Adaboost];
       
    end   %到这里结束每个子空间的数据训练
    
   %%   
    % Apply to stack   应用堆栈的思想把所有独立的分类器堆叠起来以形成最终的预测
    m_w = size(weight,1);
    for i=1:m_w
       w1 = weight(i,:);
       P = ensemble(:,1)*w1(1,1) + ensemble(:,2)*w1(1,2) + ensemble(:,3)*w1(1,3);   %这个P就是最终的预测。可以作为输出。
       Final_Accuracy_with_ensebleL = mean(P == testY) * 100; % Final accuracy      %这个就是最终预测的精度。 
       ACC_w1_all = [ACC_w1_all; Final_Accuracy_with_ensebleL];
     end
   [ACC_svml_tt,indx1] = max(max(ACC_w1_all));
   fprintf('\nproposed with binary+svml Accuracy(train&test): %f\n', ACC_svml_tt);
   tr2 = [tr2;ACC_svml_tt]; 
end

A_final = sum(tr2)/10;
toc
% 一些参数说明：
%整个函数在10次循环之下构建的。
% i：跑1-10次 前五次85.7143 ，后五次88.0952 。 平均值 86.9048
% S：S的1:3代表 代表三个子空间的
%iter ：代表每一个子空间训练的次数，设置为3 可能提前结束，也可能3次不够，主要依据是是否还有错分样本，或者是预先设置的错误率。
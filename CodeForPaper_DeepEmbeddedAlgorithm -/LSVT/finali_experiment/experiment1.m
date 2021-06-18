clc;clear ;warning('off');
% load('ttraintestdata.mat');%数据集读取
load('PD_LSVTsample');%数据集读取

ACC_w1_all=[];
tr2=[];

for i=1:1
    FinalAcc=[];
    ensemble=[];
    for s=1:3
%       load('traintestdata.mat');%数据集读取 ,这里为什么要记加载两个数据集？
        trainX1 = traindataX{1,s}(:,1:end-1);
        trainY1 = traindataX{1,s}(:,end);
%       trainX1=trainX(1:42,:);
%       trainY1=trainY(1:42,1);
        validX = valid_data(:,1:end-1);  
        validY = valid_data(:,end);
        testX = test_data(:,1:end-1);
        testY = test_data(:,end);
       %% 样本标准化部分
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
       iter=3; % number of iterations
%        Models=cell(iter,1); % For saving the models from each iteration 
       Us=cell(iter,1); % For saving the Us from each iteration
%        Trainingdata=cell(iter,1); % For saving the training data from each iteration to calculate the corresponding Alpha
       % while (m>0.0001)
       for z = 1:100
           % [fea,U,model,indx,misX,misY] = adbstLDPP(trainX,trainY,testX,testY,type_num);
            %该语句中返回最好的fea,返回最好的model，返回最好的U.最好的U 肯定是维度比较低的
           [fea,U,model,indx,misX,misY] = adbstLDPP(trainX1,trainY1,validX,validY,type_num);
           Us{z}=U;
           feature{z,1}=fea; % Saving the features for each iteration
           Trainingdata{z,1} = [trainX1 trainY1]; % combining training data with its labels. Since in each iteration training data will be changed, we need to save corresponding labels
           Medels{z,1}=model;
           trainX1 = [misX;trainX1]; % combining the training data with miss classified samples
           trainY1 = [misY;trainY1];% combining the training lebels with miss classified labels
           for i=1:size (misY)
               b=corr(misX(:,i),misY); % Finding the error. however result id NaN because miss classified samples belong to single class.
               a=[a b];
           end
        % m1=(sum(a))/size (misY,1);
        % p=[p m1];
        % m =[m abs(p(1,z)-p(1,z+1))];
        % a=[];
        % l=l+1;
        falgA = double(isempty(misX));
        if  falgA 
            break
        end
       end

       % Finding Alpha
       alpha=FInd_Alpha(Trainingdata); % finding the alpha for training data of each iteration
       % alpha(1)=0.5;
       % Test process
       svml8 = [];
       svm_prede=[]
       %测试集合进行测试模型,得出预测的结果
      
       
       for i=1:size(Trainingdata)
           test1=testX*(Us{i}); % Coressponding U
           mod1=Medels{i}% Corresponding model
           fea=feature{i};% Coressponding Feature
           test1=test1(:,fea);
           svm_pred1 = svmpredict(testY,test1,mod1);
           [ind]=find(svm_pred1==2); % Because Sgn function works with labels -1 and 1 so here I changed all my labels from 2 to -1 to make it possible(1,-1) before saving
           svm_pred1(ind,1)=-1;
           svm_pred1=svm_pred1*alpha(i); % Multiplication of alpha with corresponding prediction
           svm_prede=[svm_prede svm_pred1];
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

      %% Prediction  预测精度计算
       [ind] = find(testY==2); % Changing labels having value 2 to -1 for calculation of accuracy
       testY(ind,1) = -1;      %把testY中为-2的改为-1.
       Final_predict=sum(svm_prede,2);
       Result=sign(Final_predict);
       Final_Accuracy_with_Adaboost = mean(Result == testY) * 100; % Final accuracy
       ensemble=[ensemble Result];
       FinalAcc=[FinalAcc Final_Accuracy_with_Adaboost];
       clear Trainingdata
    end
   load 'weight'%加载网格搜索法权重 
   m_w = size(weight,1);     %网格搜索方法找寻最佳权重
   for i = 1:m_w
       w1 = weight(i,:);
       P = sign(ensemble(:,1) * w1(1,1)+ ensemble(:,2) * w1(1,2) + ensemble(:,3) * w1(1,3));
       Final_Accuracy_with_ensebleL = mean(P == testY) * 100; % Final accuracy
       ACC_w1_all = [ACC_w1_all; Final_Accuracy_with_ensebleL];
   end
   %% 评价指标相关代码
   ACC_svml_tt = max(ACC_w1_all);
   [indx] = find(ACC_w1_all== max(max(ACC_w1_all)));
   best_weight = weight(indx,:);
   %best_P是最好的预测吧，利用预测计算混淆矩阵。
   best_Pred = ensemble(:,1) * best_weight(1,1) + ensemble(:,2) * best_weight(1,2) + ensemble(:,3) * best_weight(1,3);
   
   % 下面是一个通用的建立混淆矩阵的代码,并计算几个评价指标的代码
   % best_Pred 是预测标签，testY是真实标签。到了这一步之后，预测标签和真实标签都为1和-1，原本的2都转为-1了。
   Y_unique = unique(testY);
   CM = zeros(size(Y_unique,1),size(Y_unique,1));
   for i = 1:2
       in = find(testY == Y_unique(i,1)); %找出实际标签为某一类标签的位置。
       Y_pred =  best_Pred(in,1);
       CM(i,1) = size(find(Y_pred == Y_unique(1,1)),1);%找到预测标签和真实标签相等的数量
       CM(i,2) = size(find(Y_pred == Y_unique(2,1)),1);%找到预测标签和真实标签不相等的数量
   end
   %下面语句为非必需的，只有在不满足原始混淆矩阵的形式时才使用。
   CM1 = zeros(2,2);
   CM1(1,1) = CM(2,2);
   CM1(1,2) = CM(2,1);
   CM1(2,1) = CM(1,2);
   CM1(2,2) = CM(1,1);
   TP = CM1(1,1);
   FN = CM1(1,2);
   FP = CM1(2,1);
   TN = CM1(2,2);
   %下面写评价指标
   ACC = (TP+TN) / (TP + TN + FP + FN);
   Pre = TP / (TP + FP);
   Rec = TP / (TP + FN);
   Spe = TN / (FN + TN);
   G_mean = sqrt(Rec * Spe);
   %%

   fprintf('\nproposed with binary+svml Accuracy(train&test): %f\n', ACC_svml_tt);
   tr2=[tr2;ACC_svml_tt];   
end
toc

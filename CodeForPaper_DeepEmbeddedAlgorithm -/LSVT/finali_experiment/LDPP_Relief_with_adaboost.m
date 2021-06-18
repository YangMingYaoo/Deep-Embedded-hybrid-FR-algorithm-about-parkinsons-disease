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
for i=1:1
    FinalAcc=[];
    ensemble=[];
    for s=1:3
        load('traintestdata.mat');%数据集读取 ,这里为什么要记加载两个数据集？
        trainX1=train_x{i,s}(:,1:end-1);
        trainY1=train_x{i,s}(:,end);
%       trainX1=trainX(1:42,:);
%       trainY1=trainY(1:42,1);
        validX=trainX(43:84,:);  %这里为何要从第43行开始采集呢？
        validY=trainY(43:84,1);
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
       Models=cell(iter,1); % For saving the models from each iteration 
       Us=cell(iter,1); % For saving the Us from each iteration
       Trainingdata=cell(iter,1); % For saving the training data from each iteration to calculate the corresponding Alpha
       % while (m>0.0001)
       for z=1:10
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

       for i=1:iter;
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
       [ind]=find(testY==2); % Changing labels having value 2 to -1 for calculation of accuracy
       testY(ind,1)=-1;      %把testY中为-2的改为-1.
       Final_predict=sum(svm_prede,2);
       Result=sign(Final_predict);
       Final_Accuracy_with_Adaboost = mean(Result == testY) * 100 % Final accuracy
       ensemble=[ensemble Result];
       FinalAcc=[FinalAcc Final_Accuracy_with_Adaboost];

    end
   
   m_w=size(weight,1);     %网格搜索方法找寻最佳权重
   for i=1:m_w
       w1=weight(i,:);
       P=ensemble(:,1)*w1(1,1)+ensemble(:,2)*w1(1,2)+ensemble(:,3)*w1(1,3);
       Final_Accuracy_with_ensebleL = mean(P == testY) * 100; % Final accuracy
       ACC_w1_all=[ACC_w1_all; Final_Accuracy_with_ensebleL];
   end
   
   [ACC_svml_tt,indx]=max(max(ACC_w1_all));
   fprintf('\nproposed with binary+svml Accuracy(train&test): %f\n', ACC_svml_tt);
   tr2=[tr2;ACC_svml_tt];   
end
toc

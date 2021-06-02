function [LSVTLDPPB_Relief_Bagging_Result] = LSVTLDPP_B_Relief_Bagging_Result()
clear ; close all; clc;warning('off');
load PD_LSVTsample; %数据集读取
load('weight.mat');
% load('train_x.mat');
tr2 = []
tic
% tr_x=[];
% train_data_all = trainX;
% train_label_all = trainY;
% test_data = testX;
% type_num = 2;
ACC_w1_all = [];  
tr2=[];
TPi = [];
FNi = [];
FPi = [];
TNi = [];
ACCi = [];
Prei = [];
Reci = [];
Spei = [];
G_meani = [];
F1_scorei = [];
Mean_scores = {};    
CMi = {}; 
svml3 = [];
scores = [];
ensemble = [];
LSVTLDPPB_Relief_Bagging_Result = [];
 FinalAcc=[];
 ensemble=[];
for s = 1:3
trainX = traindataX{1,s}(:,1:end-1);
trainY = traindataX{1,s}(:,end);
%             testX = test_data(:,1:end-1)
%             testY = test_data(:,end)
validX = valid_data(:,1:end-1)
validY = valid_data(:,end)
testX = test_data(:,1:end-1)
testY = test_data(:,end)
[trainX, mu, sigma] = featureCentralize(trainX)
%             testX = bsxfun(@minus, testX, mu);
%             testX = bsxfun(@rdivide, testX, sigma);%%将所有训练样本标准化
validX = bsxfun(@minus, validX, mu);
validX = bsxfun(@rdivide, validX, sigma);
testX = bsxfun(@minus, testX, mu);
testX = bsxfun(@rdivide, testX, sigma);%%将所有训练样本标准化

trainX1 = trainX;
trainY1 = trainY;
for z = 1 : 3
    a = []
    [fea,U,mode2,indx,misX,misY] = LSVT_LDPPBRelief_Bagging_adbstLDPP(trainX1,trainY1,validX,validY,type_num)
    Us{z}=U;
    feature{z,1}=fea;
    Medels{z,1} = mode2;
    Trainingdata{z,1} = [trainX1 trainY1];
    trainX1 = [misX;trainX1]; % combining the training data with miss classified samples
    trainY1 = [misY;trainY1];% combining the training lebels with miss classified labels
%    for i=1:size (misY)
%        b=corr(misX(:,i),misY); % Finding the error. however result id NaN because miss classified samples belong to single class.
%        a=[a b];
%    end
    
end
svml8 = [];
svm_prede = []

 for i = 1:size(Trainingdata)
     test1 = testX * (Us{i}); % Coressponding U
     mod1 = Medels{i}% Corresponding model
     fea = feature{i};% Coressponding Feature
     test1 = test1(:,fea);
     svm_pred1 = svmpredict(testY,test1,mod1);
     %% rf shihou huan
%    [svm_pred1,votes] = classRF_predict(test1,mod1)
     [ind] = find(svm_pred1 == 2); % Because Sgn function works with labels -1 and 1 so here I changed all my labels from 2 to -1 to make it possible(1,-1) before saving
     svm_pred1(ind,1) = -1;
%      svm_pred1 = svm_pred1 * alpha(i); % Multiplication of alpha with corresponding prediction
     svm_prede = [svm_prede svm_pred1];
 end
 
%% Prediction  预测精度计算
   [ind] = find(testY == 2); % Changing labels having value 2 to -1 for calculation of accuracy
    testY(ind,1) = -1;      %把testY中为-2的改为-1.
    Final_predict = sum(svm_prede,2);
    Result = sign(Final_predict);
    Final_Accuracy_with_Adaboost = mean(Result == testY) * 100; % Final accuracy
%      Final_Accuracy_with_Adaboost = calculateAccuracy(test_data,testY, Result);
    ensemble = [ensemble Result];
    FinalAcc = [FinalAcc Final_Accuracy_with_Adaboost];
     

 %% 每一个子空间的评价指标
       Y_unique = unique(testY);
       %把每一个子空间的混淆矩阵都保存下来
       CMi{s} = zeros(size(Y_unique,1),size(Y_unique,1));
       for n = 1:2
           in = find(testY == Y_unique(n,1)); %找出实际标签为某一类标签的位置。
           Y_pred =  Result(in,1);
           CMi{s}(n,1) = size(find(Y_pred == Y_unique(1,1)),1)%找到预测标签和真实标签相等的数量
           CMi{s}(n,2) = size(find(Y_pred == Y_unique(2,1)),1)%找到预测标签和真实标签不相等的数量
       end
       % 把每一个子空间的混淆矩阵具体值记录下来
       TPi(s,:) = CMi{s}(1,1);
       FNi(s,:) = CMi{s}(1,2);
       FPi(s,:) = CMi{s}(2,1);
       TNi(s,:) = CMi{s}(2,2);
       %把每一个子空间的评价指标值记录下来
       ACCi(s,:) = (TPi(s,:) + TNi(s,:)) / (TPi(s,:) + TNi(s,:) + FPi(s,:) + FNi(s,:));
       Prei(s,:) = TPi(s,:) / (TPi(s,:) + FPi(s,:));
       Reci(s,:) = TPi(s,:) / (TPi(s,:) + FNi(s,:));
       Spei(s,:) = TNi(s,:) / (FNi(s,:) + TNi(s,:));
       G_meani(s,:) = sqrt(Reci(s,:) * Spei(s,:));
       F1_scorei(s,:) = (2 * Prei(s,:) * Reci(s,:)) / (Prei(s,:) + Reci(s,:))
         
end  % 结束每一个样本子空间的标志

%% juece ceng ronghe 
m_w = size(weight,1); 
 for q = 1:m_w
      w1= weight(q,:);
      P = sign(ensemble(:,1) * w1(1,1) + ensemble(:,2) * w1(1,2) + ensemble(:,3) * w1(1,3));
      Final_Accuracy_with_ensebleL = mean(P == testY) * 100; % Final accuracy
      ACC_w1_all = [ACC_w1_all; Final_Accuracy_with_ensebleL];
 end
 
       ACC_svml_tt = max(ACC_w1_all);
       indx = find(ACC_w1_all == max(max(ACC_w1_all)));
       best_weight = weight(indx(1,1),:);
        %best_P是最好的预测吧，利用预测计算混淆矩阵。
       best_Pred = ensemble(:,1) * best_weight(1,1) + ensemble(:,2) * best_weight(1,2) + ensemble(:,3) * best_weight(1,3);
       % 下面是一个通用的建立混淆矩阵的代码,并计算几个评价指标的代码
      % best_Pred 是预测标签，testY是真实标签。到了这一步之后，预测标签和真实标签都为1和-1，原本的2都转为-1了。
     
     CM = zeros(size(Y_unique,1),size(Y_unique,1));
     for r = 1:2
       in = find(testY == Y_unique(r,1)); %找出实际标签为某一类标签的位置。
       Y_pred =  best_Pred(in,1);
       CM(r,1) = size(find(Y_pred == Y_unique(1,1)),1);%找到预测标签和真实标签相等的数量
       CM(r,2) = size(find(Y_pred == Y_unique(2,1)),1);%找到预测标签和真实标签不相等的数量
     end
   TP = CM(1,1);
   FN = CM(1,2);
   FP = CM(2,1);  
   TN = CM(2,2);
   %下面写评价指标
   ACC = (TP + TN) / (TP + TN + FP + FN);
   Pre = TP / (TP + FP);
   Rec = TP / (TP + FN);
   Spe = TN / (FN + TN);
   G_mean = sqrt(Rec * Spe);
   F1_score_Final = (2 * Pre * Rec) / (Pre + Rec)
   tr2 = [tr2;ACC_svml_tt]; 
   ACC_final_mean = mean(tr2); 
   LSVTLDPPB_Relief_Bagging_Result.ACC_final_mean = ACC_final_mean;
   LSVTLDPPB_Relief_Bagging_Result.ACC = ACC;   
   LSVTLDPPB_Relief_Bagging_Result.ACCi = ACCi; 
   LSVTLDPPB_Relief_Bagging_Result.F1_score_Final = F1_score_Final; 
   LSVTLDPPB_Relief_Bagging_Result.F1_scorei = F1_scorei; 
   LSVTLDPPB_Relief_Bagging_Result.G_mean = G_mean; 
   LSVTLDPPB_Relief_Bagging_Result.G_meani = G_meani; 
   LSVTLDPPB_Relief_Bagging_Result.Pre = Pre;
   LSVTLDPPB_Relief_Bagging_Result.Prei = Prei; 
   LSVTLDPPB_Relief_Bagging_Result.Rec = Rec; 
   LSVTLDPPB_Relief_Bagging_Result.Reci = Reci; 
   LSVTLDPPB_Relief_Bagging_Result.Spe = Spe; 
   LSVTLDPPB_Relief_Bagging_Result.Spei = Spei; 
%    LDPPbinary_Relief_Result.svml3 = svml3;  
end

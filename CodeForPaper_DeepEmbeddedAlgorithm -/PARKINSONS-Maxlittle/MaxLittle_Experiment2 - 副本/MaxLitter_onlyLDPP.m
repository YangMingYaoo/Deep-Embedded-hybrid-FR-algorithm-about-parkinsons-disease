%% 
clc;clear ;warning('off');
tic
final_eval= {};
mean_final_eval = {};
iter = 1
for C = 1:iter
    [MaxLittlerProposed_H_experiment1_s2] = onlyLDPPB()
    final_eval{C} = MaxLittlerProposed_H_experiment1_s2;
end
mean_final_eval.ACC = (final_eval{1}.ACC + final_eval{2}.ACC + final_eval{3}.ACC+final_eval{4}.ACC + final_eval{5}.ACC + final_eval{6}.ACC+final_eval{7}.ACC + final_eval{8}.ACC + final_eval{9}.ACC + final_eval{10}.ACC)/10
mean_final_eval.ACCi = (final_eval{1}.ACCi + final_eval{2}.ACCi + final_eval{3}.ACCi + final_eval{4}.ACCi + final_eval{5}.ACCi + final_eval{6}.ACCi + final_eval{7}.ACCi + final_eval{8}.ACCi + final_eval{9}.ACCi + final_eval{10}.ACCi)/10
mean_final_eval.Pre = (final_eval{1}.Pre + final_eval{2}.Pre + final_eval{3}.Pre + final_eval{4}.Pre + final_eval{5}.Pre + final_eval{6}.Pre + final_eval{7}.Pre + final_eval{8}.Pre + final_eval{9}.Pre + final_eval{10}.Pre)/10
mean_final_eval.Prei = (final_eval{1}.Prei + final_eval{2}.Prei + final_eval{3}.Prei + final_eval{4}.Prei + final_eval{5}.Prei + final_eval{6}.Prei + final_eval{7}.Prei + final_eval{8}.Prei + final_eval{9}.Prei + final_eval{10}.Prei)/10
mean_final_eval.Rec = (final_eval{1}.Reci + final_eval{2}.Reci + final_eval{3}.Reci + final_eval{4}.Reci + final_eval{5}.Reci + final_eval{6}.Reci + final_eval{7}.Reci + final_eval{8}.Reci + final_eval{9}.Reci + final_eval{10}.Reci)/10
mean_final_eval.Reci = (final_eval{1}.Reci + final_eval{2}.Reci + final_eval{3}.Reci + final_eval{4}.Reci + final_eval{5}.Reci + final_eval{6}.Reci + final_eval{7}.Reci + final_eval{8}.Reci + final_eval{9}.Reci + final_eval{10}.Reci)/10
mean_final_eval.TNR = (final_eval{1}.TNR + final_eval{2}.TNR + final_eval{3}.TNR + final_eval{4}.TNR + final_eval{5}.TNR + final_eval{6}.TNR + final_eval{7}.TNR + final_eval{8}.TNR + final_eval{9}.TNR + final_eval{10}.TNR)/10
mean_final_eval.TNRi = (final_eval{1}.TNRi + final_eval{2}.TNRi + final_eval{3}.TNRi + final_eval{4}.TNRi + final_eval{5}.TNRi + final_eval{6}.TNRi + final_eval{7}.TNRi + final_eval{8}.TNRi + final_eval{9}.TNRi + final_eval{10}.TNRi)/10
mean_final_eval.Spe = (final_eval{1}.Spe + final_eval{2}.Spe + final_eval{3}.Spe + final_eval{4}.Spe + final_eval{5}.Spe + final_eval{6}.Spe + final_eval{7}.Spe + final_eval{8}.Spe + final_eval{9}.Spe + final_eval{10}.Spe)/10
mean_final_eval.Spei = (final_eval{1}.Spei + final_eval{2}.Spei + final_eval{3}.Spei + final_eval{4}.Spei + final_eval{5}.Spei + final_eval{6}.Spei + final_eval{7}.Spei + final_eval{8}.Spei + final_eval{9}.Spei + final_eval{10}.Spei)/10
mean_final_eval.G_mean = (final_eval{1}.G_mean + final_eval{2}.G_mean + final_eval{3}.G_mean + final_eval{4}.G_mean + final_eval{5}.G_mean + final_eval{6}.G_mean + final_eval{7}.G_mean + final_eval{8}.G_mean + final_eval{9}.G_mean + final_eval{10}.G_mean)/10
mean_final_eval.G_meani = (final_eval{1}.G_meani + final_eval{2}.G_meani + final_eval{3}.G_meani + final_eval{4}.G_meani + final_eval{5}.G_meani + final_eval{6}.G_meani + final_eval{7}.G_meani + final_eval{8}.G_meani + final_eval{9}.G_meani + final_eval{10}.G_meani)/10
mean_final_eval.F1_score = (final_eval{1}.F1_score + final_eval{2}.F1_score + final_eval{3}.F1_score + final_eval{4}.F1_score + final_eval{5}.F1_score+ final_eval{6}.F1_score + final_eval{7}.F1_score + final_eval{8}.F1_score + final_eval{9}.F1_score + final_eval{10}.F1_score)/10
mean_final_eval.F1_scorei = (final_eval{1}.F1_scorei + final_eval{2}.F1_scorei + final_eval{3}.F1_scorei + final_eval{4}.F1_scorei + final_eval{5}.F1_scorei + final_eval{6}.F1_scorei + final_eval{7}.F1_scorei + final_eval{8}.F1_scorei + final_eval{9}.F1_scorei + final_eval{10}.F1_scorei)/10
toc
%% 

function [onlyLDPPB_result] = onlyLDPPB()
clear ; close all; clc;warning('off');   
load 'maxlittle_sample4'; %数据集读取
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
TNR = [];
TNRi = [];
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
onlyLDPPB_result = [];

for s = 1:3
trainX = traindataX{1,s}(:,2:end-1);
trainY = traindataX{1,s}(:,end);
%             testX = test_data(:,1:end-1)
%             testY = test_data(:,end)
validX = valid_data(:,2:end-1)
validY = valid_data(:,end)
testX = test_data(:,2:end-1)
testY = test_data(:,end)
[trainX, mu, sigma] = featureCentralize(trainX)
%             testX = bsxfun(@minus, testX, mu);
%             testX = bsxfun(@rdivide, testX, sigma);%%将所有训练样本标准化
validX = bsxfun(@minus, validX, mu);
validX = bsxfun(@rdivide, validX, sigma);
testX = bsxfun(@minus, testX, mu);
testX = bsxfun(@rdivide, testX, sigma);%%将所有训练样本标准化

for iter=1:1
kk=2;
mukgamma=[];
mean_svml8_max=0;
for igamma=1:9
    for imu=1:9
        method = [];
        method.mode = 'ldpp_u';
        method.mu = 0.00001 * power(10,imu);
        method.gamma = 0.00001 * power(10,igamma);  
        method.M = 200;
        method.labda2 = 0.001;%取[0.0001,0.001,...,1000,10000]
        method.ratio_b = 0.9;
        method.ratio_w = 0.9;
        method.weightmode = 'binary';
        method.knn_k = 5;
        svml8 = [];
        
%             trainX = train_data_all(:,1:end);
%             testX=test_data(:,1:end);
%             trainY=train_label_all;
%           %% 给样本添加噪声
%             train_cqX = traindataX{iter,i}(:,1:end-1);
%             train_cqY = traindataX{iter,i}(:,end);
% %             testX = test_data(:,1:end-1)
% %             testY = test_data(:,end)
%             validX = valid_data(:,1:end-1)
%             validY = valid_data(:,end)
%             [train_cqX, mu, sigma] = featureCentralize(train_cqX)
% %             testX = bsxfun(@minus, testX, mu);
% %             testX = bsxfun(@rdivide, testX, sigma);%%将所有训练样本标准化
%             validX = bsxfun(@minus, validX, mu);
%             validX = bsxfun(@rdivide, validX, sigma);
            U = featureExtract2(trainX,trainY,method,type_num);
            U1_all{1,s} = U;
            for ik = 1:floor(size(trainX,2)/2)
                method.K = kk * ik;
                mukgamma=[mukgamma;[imu ik igamma]];    
                trainZ=projectData(trainX, U, method.K);
                validZ = projectData(validX, U, method.K);
       
                % SVM 高斯 
                model = svmtrain(trainY,trainZ,'-s 0 -t 2');
                 svm_pred1 = svmpredict(validY,validZ,model);
                svml8(ik) = mean(svm_pred1 == validY) * 100;
                %% RF 
%             model = classRF_train(trainZ,trainY,'ntree',300);
%             [svm_pred1,votes] = classRF_predict(validZ,model);
%             svml8(ik) = mean(svm_pred1 == validY) * 100
    
            %% SVM线性  
%                 model = svmtrain(trainY,trainZ,'-s 0 -t 0');%%使用所有变换后的训练集训练模型
%                 svm_pred1 = svmpredict(validY,validZ,model);
%                 svml8(ik) = mean(svm_pred1 == validY) * 100;
            end
            [acc_svml_max,indx2] = max(svml8);
               Accuracy(igamma,imu) = acc_svml_max;
               best_svml_kk = kk * indx2; 
               bestK(igamma,imu) = best_svml_kk;
    end
 end
%         U1=(U1_all{1,1}+U1_all{1,2}+U1_all{1,3})/3;
        [loc_x,loc_y] = find(Accuracy==max(max(Accuracy)));%找到最大值的位置
        mean_svml8_max = max(max(Accuracy)); 
        best_svml_kk = bestK(loc_x(1),loc_y(1));
        method.mu = 0.00001 * power(10,loc_y(1));%取[0.0001,0.001,...,1000,10000]
        method.gamma = 0.00001 * power(10,loc_x(1));
        U = featureExtract2(trainX,trainY,method,type_num);
        trainZ1 = projectData(trainX, U, best_svml_kk);
        testZ1 = projectData(testX, U, best_svml_kk);
end

% SVM 线性
%     model = svmtrain(trainY,train,'-s 0 -t 0');
%     svm_pred3 = svmpredict(testY,test,model);
%     svml6= mean(svm_pred3 == testY) * 100 ;
    
      % SVM 高斯
        model = svmtrain(trainY,trainZ1,'-s 0 -t 2');
        svm_pred3 = svmpredict(testY,testZ1,model);
        svml6= mean(svm_pred3 == testY) * 100 ;
    
    % RF
%         model = classRF_train(trainZ1,trainY,'ntree',300)
%         [svm_pred3,votes] = classRF_predict(testZ1,model)
%         svml6 = mean(svm_pred3 == testY) * 100 

%% 每一个子空间的评价 
 % 因为后面程序按照标签为[-1,1]写的，所以这里把2变成-1.
        [ind] = find(svm_pred3 == 2); % Because Sgn function works with labels -1 and 1 so here I changed all my labels from 2 to -1 to make it possible(1,-1) before saving
        svm_pred3(ind,1) = -1;
        % 把每一个子空间的预测保存下来
        ensemble = [ensemble svm_pred3];   
        [ind] = find(testY==2); % Changing labels having value 2 to -1 for calculation of accuracy
        testY(ind,1) = -1;      %把testY中为-2的改为-1.
     %% 每一个子空间的评价指标
       Y_unique = unique(testY);
       %把每一个子空间的混淆矩阵都保存下来
       CMi{s} = zeros(size(Y_unique,1),size(Y_unique,1));
       for n = 1:2
           in = find(testY == Y_unique(n,1)); %找出实际标签为某一类标签的位置。
           Y_pred =  svm_pred3(in,1);
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
       Spei(s,:) = TNi(s,:) / (FPi(s,:) + TNi(s,:));
%        TNRi(s,:) = TNi(s,:) / (FPi(s,:) + TNi(s,:));
       G_meani(s,:) = sqrt(Reci(s,:) * Spei(s,:));
       F1_scorei(s,:) = (2 * Prei(s,:) * Reci(s,:)) / (Prei(s,:) + Reci(s,:))
      
      
end  %每一个子空间结束的标志
%% 三层空间的评价
%% juece ceng ronghe 
m_w = size(weight,1); 
 for q = 1:m_w
      w1= weight(q ,:);
      P = sign(ensemble(:,1) * w1(1,1) + ensemble(:,2) * w1(1,2) + ensemble(:,3) * w1(1,3));
      Final_Accuracy_with_ensebleL = mean(P == testY) * 100; % Final accuracy
      ACC_w1_all = [ACC_w1_all; Final_Accuracy_with_ensebleL];
 end
       ACC_svml_tt = max(ACC_w1_all);
       indx = find(ACC_w1_all == max(max(ACC_w1_all)));
       best_weight = weight(indx(1,1),:);
        %best_P是最好的预测吧，利用预测计算混淆矩阵。
       best_Pred = sign(ensemble(:,1) * best_weight(1,1) + ensemble(:,2) * best_weight(1,2) + ensemble(:,3) * best_weight(1,3));
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
   TNR = TN / (FP + TN);
   Spe = TN / (FP + TN);
  
   G_mean = sqrt(Rec * Spe);
   F1_score = (2 * Pre * Rec) / (Pre + Rec)
   tr2 = [tr2;ACC_svml_tt]; 
   ACC_final_mean = mean(tr2); 
   onlyLDPPB_result.ACC = ACC;   
   onlyLDPPB_result.ACCi = ACCi; 
   onlyLDPPB_result.Pre = Pre;
   onlyLDPPB_result.Prei = Prei
   onlyLDPPB_result.Rec = Rec; 
   onlyLDPPB_result.Reci = Reci; 
%    onlyLDPPB_result.TNR = TNR;
%    onlyLDPPB_result.TNRi = TNRi
   onlyLDPPB_result.Spe = Spe; 
   onlyLDPPB_result.Spei = Spei;
   onlyLDPPB_result.G_mean = G_mean; 
   onlyLDPPB_result.G_meani = G_meani; 
   onlyLDPPB_result.F1_score = F1_score; 
   onlyLDPPB_result.F1_scorei = F1_scorei; ; 
end

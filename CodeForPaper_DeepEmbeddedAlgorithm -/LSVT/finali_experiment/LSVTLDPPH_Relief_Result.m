function [LSVTLDPPH_Relief_Result] = LSVTLDPPH_Relief_Result()
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
Evaluation_index_i = [];
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
LSVTLDPPH_Relief_Result = [];
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


for iter=1:1
kk=5;
mukgamma=[];
mean_svml8_max=0;
for igamma=1:9
    for imu=1:9
        method = [];
        method.mode = 'ldpp_u';
        method.mu=0.00001*power(10,imu);
        method.gamma=0.00001*power(10,igamma);  
        method.M = 200;
        method.labda2 = 0.001;%取[0.0001,0.001,...,1000,10000]
        method.ratio_b = 0.9;
        method.ratio_w = 0.9;
        method.weightmode = 'heatkernel';
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
            for ik = 1:floor(size(trainX,2)/5)
                method.K = kk * ik;
                mukgamma=[mukgamma;[imu ik igamma]];    
                trainZ=projectData(trainX, U, method.K);
                validZ = projectData(validX, U, method.K);
       
                % SVM 高斯 
%                 model = svmtrain(trainY,trainZ,'-s 0 -t 2');
%                  svm_pred1 = svmpredict(validY,validZ,model);
%                 svml8(ik) = mean(svm_pred1 == validY) * 100;
                %% RF 
%             model = classRF_train(trainZ,trainY,'ntree',300)
%             [svm_pred1,votes] = classRF_predict(validZ,model)
%             svml8(ik) = mean(svm_pred1 == validY) * 100
    
            %% SVM线性  
                model = svmtrain(trainY,trainZ,'-s 0 -t 0');%%使用所有变换后的训练集训练模型
                svm_pred1 = svmpredict(validY,validZ,model);
                svml8(ik) = mean(svm_pred1 == validY) * 100;
            end
        end
%         U1=(U1_all{1,1}+U1_all{1,2}+U1_all{1,3})/3;
        [loc_x,loc_y] = find(svml8==max(max(svml8)));%找到最大值的位置
%         [loc_x,loc_y]=max(mean(svml8));%找到最大值的位置
        if  max(max(svml8))>mean_svml8_max
            mean_svml8_max=max(max(svml8));
            U_svml_best=U1_all;
            best_svml_kk = kk*loc_y(1,1);
            best_svml_mu = 0.00001*power(10,imu);%取[0.0001,0.001,...,1000,10000]
            best_svml_gamma = 0.00001*power(10,igamma);
        end
    end
end


method.mu = best_svml_mu;
method.gamma = best_svml_gamma;
method.K = best_svml_kk;
U = featureExtract2(trainX,trainY,method,type_num);
trainZ1 = projectData(trainX, U, method.K);
validZ1 = projectData(validX, U, method.K);
% relief
[fea] = relieff(trainZ1,trainY, 5)
svml7 = [];
for p = 1:floor(size(trainZ1,2)/5) %  size(train,2) 计算train矩阵的列数
  K = kk * p ;
    trainZ2 = trainZ1(:,fea(:,1:K));   %在排列好序的特征中选出权重较大的特征进行实验，依次选出进行试验。
    validZ2 = validZ1(:,fea(:,1:K)); 
    %到了这里 数据集先试用LDPP ，在使用relief. 对LDPP变换出来的新特征在进行特征选择
    %% SVM 线性
    model = svmtrain(trainY,trainZ2,'-s 0 -t 0');
    svm_pred2 = svmpredict(validY,validZ2,model);
    svml7(p) = mean(svm_pred2 == validY) * 100 ;
    
  %% SVM 高斯
%         model = svmtrain(trainY,trainZ2,'-s 0 -t 2');
%         svm_pred2 = svmpredict(validY,validZ2,model);
%        svml7(p) = mean(svm_pred2 == validY) * 100 ;
    %% RF
%         model = classRF_train(trainZ2,trainY,'ntree',300)
%         [svm_pred2,votes] = classRF_predict(validZ2,model)
%        svml7(p) = mean(svm_pred2 == validY) * 100 
        
    
 end

[~,index] = max( svml7);
best_fea = fea(:,1:index);
% 
[trainX, mu, sigma] = featureCentralize(trainX)
testX = bsxfun(@minus, testX, mu);
testX = bsxfun(@rdivide, testX, sigma);%%将所有训练样本标准化

train = projectData(trainX, U, method.K);
train = train(:,best_fea);
test =  projectData(testX, U, method.K);
test = test(:,best_fea);

%test 
 %% SVM 线性
    model = svmtrain(trainY,train,'-s 0 -t 0');
    svm_pred3 = svmpredict(testY,test,model);
    svml6= mean(svm_pred3 == testY) * 100 ;
    
      %% SVM 高斯
%         model = svmtrain(trainY,train,'-s 0 -t 2');
%         svm_pred3 = svmpredict(testY,test,model);
%         svml6= mean(svm_pred3 == testY) * 100 ;
    
    %% RF
%         model = classRF_train(train,trainY,'ntree',300)
%         [svm_pred3,votes] = classRF_predict(test,model)
%         svml6= mean(svm_pred3 == testY) * 100 

%% ping jia 
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
       Spei(s,:) = TNi(s,:) / (FNi(s,:) + TNi(s,:));
       G_meani(s,:) = sqrt(Reci(s,:) * Spei(s,:));
       F1_scorei(s,:) = (2 * Prei(s,:) * Reci(s,:)) / (Prei(s,:) + Reci(s,:))
         
end

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
   LSVTLDPPH_Relief_Result.ACC_final_mean = ACC_final_mean;
   LSVTLDPPH_Relief_Result.ACC = ACC;   
   LSVTLDPPH_Relief_Result.ACCi = ACCi; 
   LSVTLDPPH_Relief_Result.F1_score_Final = F1_score_Final; 
   LSVTLDPPH_Relief_Result.F1_scorei = F1_scorei; 
   LSVTLDPPH_Relief_Result.G_mean = G_mean; 
   LSVTLDPPH_Relief_Result.G_meani = G_meani; 
   LSVTLDPPH_Relief_Result.Pre = Pre;
   LSVTLDPPH_Relief_Result.Prei = Prei; 
   LSVTLDPPH_Relief_Result.Rec = Rec; 
   LSVTLDPPH_Relief_Result.Reci = Reci; 
   LSVTLDPPH_Relief_Result.Spe = Spe; 
   LSVTLDPPH_Relief_Result.Spei = Spei; 
%    LDPPbinary_Relief_Result.svml3 = svml3;  
end

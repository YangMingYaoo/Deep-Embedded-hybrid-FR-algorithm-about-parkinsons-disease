% % 
% clc;clear ;warning('off');
% tic
% final_eval= {};
% mean_final_eval = {};
% iter = 1
% for C = 1:iter
%     [sakar_subject_Result] = MaxLitterProposed_experiment()
%     final_eval{C} =  sakar_subject_Result;
% end
% mean_final_eval.ACC = (final_eval{1}.ACC + final_eval{2}.ACC + final_eval{3}.ACC+final_eval{4}.ACC + final_eval{5}.ACC + final_eval{6}.ACC+final_eval{7}.ACC + final_eval{8}.ACC + final_eval{9}.ACC + final_eval{10}.ACC)/10
% mean_final_eval.ACCi = (final_eval{1}.ACCi + final_eval{2}.ACCi + final_eval{3}.ACCi + final_eval{4}.ACCi + final_eval{5}.ACCi + final_eval{6}.ACCi + final_eval{7}.ACCi + final_eval{8}.ACCi + final_eval{9}.ACCi + final_eval{10}.ACCi)/10
% mean_final_eval.Pre = (final_eval{1}.Pre + final_eval{2}.Pre + final_eval{3}.Pre + final_eval{4}.Pre + final_eval{5}.Pre + final_eval{6}.Pre + final_eval{7}.Pre + final_eval{8}.Pre + final_eval{9}.Pre + final_eval{10}.Pre)/10
% mean_final_eval.Prei = (final_eval{1}.Prei + final_eval{2}.Prei + final_eval{3}.Prei + final_eval{4}.Prei + final_eval{5}.Prei + final_eval{6}.Prei + final_eval{7}.Prei + final_eval{8}.Prei + final_eval{9}.Prei + final_eval{10}.Prei)/10
% mean_final_eval.Rec = (final_eval{1}.Rec + final_eval{2}.Rec + final_eval{3}.Rec + final_eval{4}.Rec + final_eval{5}.Rec + final_eval{6}.Rec + final_eval{7}.Rec + final_eval{8}.Rec + final_eval{9}.Rec + final_eval{10}.Rec)/10
% mean_final_eval.Reci = (final_eval{1}.Reci + final_eval{2}.Reci + final_eval{3}.Reci + final_eval{4}.Reci + final_eval{5}.Reci + final_eval{6}.Reci + final_eval{7}.Reci + final_eval{8}.Reci + final_eval{9}.Reci + final_eval{10}.Reci)/10
% mean_final_eval.Spe = (final_eval{1}.Spe + final_eval{2}.Spe + final_eval{3}.Spe + final_eval{4}.Spe + final_eval{5}.Spe + final_eval{6}.Spe + final_eval{7}.Spe + final_eval{8}.Spe + final_eval{9}.Spe + final_eval{10}.Spe)/10
% mean_final_eval.Spei = (final_eval{1}.Spei + final_eval{2}.Spei + final_eval{3}.Spei + final_eval{4}.Spei + final_eval{5}.Spei + final_eval{6}.Spei + final_eval{7}.Spei + final_eval{8}.Spei + final_eval{9}.Spei + final_eval{10}.Spei)/10
% mean_final_eval.G_mean = (final_eval{1}.G_mean + final_eval{2}.G_mean + final_eval{3}.G_mean + final_eval{4}.G_mean + final_eval{5}.G_mean + final_eval{6}.G_mean + final_eval{7}.G_mean + final_eval{8}.G_mean + final_eval{9}.G_mean + final_eval{10}.G_mean)/10
% mean_final_eval.G_meani = (final_eval{1}.G_meani + final_eval{2}.G_meani + final_eval{3}.G_meani + final_eval{4}.G_meani + final_eval{5}.G_meani + final_eval{6}.G_meani + final_eval{7}.G_meani + final_eval{8}.G_meani + final_eval{9}.G_meani + final_eval{10}.G_meani)/10
% mean_final_eval.F1_score = (final_eval{1}.F1_score + final_eval{2}.F1_score + final_eval{3}.F1_score + final_eval{4}.F1_score + final_eval{5}.F1_score+ final_eval{6}.F1_score + final_eval{7}.F1_score + final_eval{8}.F1_score + final_eval{9}.F1_score + final_eval{10}.F1_score)/10
% mean_final_eval.F1_scorei = (final_eval{1}.F1_scorei + final_eval{2}.F1_scorei + final_eval{3}.F1_scorei + final_eval{4}.F1_scorei + final_eval{5}.F1_scorei + final_eval{6}.F1_scorei + final_eval{7}.F1_scorei + final_eval{8}.F1_scorei + final_eval{9}.F1_scorei + final_eval{10}.F1_scorei)/10
% toc
% % 



function [sakar_subject_Result  ] = code1_PSDMTSR_Proposed_experiment1( )

clc; clear all; close all; warning('off');
    % load('ttraintestdata.mat');%数据集读取
load('sakar_sample');%数据集读取
ACC_w1_all=[];
tr2=[]
TPi = [];
FNi = [];
FPi = [];
TNi = [];
ACCi = [];
Prei = [];
TNRi = [];
Reci = [];
Spei = [];
G_meani = [];
F1_scorei = [];
Mean_scores = {};    
CMi = {}; 
svml3 = [];
scores = [];
ensemble = [];
sakar_subject_Result = [];

FinalAcci=[];
ensemble=[];
for i=1:1
    
 for s=1:3
%       load('traintestdata.mat');%数据集读取 ,这里为什么要记加载两个数据集？
        trainX1 = traindataX{1,s}(:,2:end-1);
        trainY1 = traindataX{1,s}(:,end);
        validX = valid_data(:,2:end-1);  
        validY = valid_data(:,end);
        testX = test_data(:,2:end-1);
        testY = test_data(:,end);
        
       %% 样本标准化部分
       [trainX1, mu, sigma] = featureCentralize(trainX1);%%将样本标准化（服从N(0,1)分布）
       testX = bsxfun(@minus, testX, mu);
       testX = bsxfun(@rdivide, testX, sigma);
       validX = bsxfun(@minus, validX, mu);
       validX = bsxfun(@rdivide, validX, sigma);    
       a=[];
       m = 1;
       p=0;
       % l=1;
       tic
       iter = 18; % number of iterations
%        Models=cell(iter,1); % For saving the models from each iteration 
       Us=cell(iter,1); % For saving the Us from each iteration
%        Trainingdata=cell(iter,1); % For saving the training data from each iteration to calculate the corresponding Alpha
       % while (m>0.0001)c
       for z = 1:iter
           % [fea,U,model,indx,misX,misY] = adbstLDPP(trainX,trainY,testX,testY,type_num);
            %该语句中返回最好的fea,返回最好的model，返回最好的U.最好的U 肯定是维度比较低的
           [fea, U, mode2, indx, misX, misY]  = sakarProposed_adbstLDPP(trainX1,trainY1,validX,validY,valid_data,type_num);
              
           Us{z} = U;
           feature{z,1} = fea; % Saving the features for each iteration
           Trainingdata{z,1} = [trainX1 trainY1]; % combining training data with its labels. Since in each iteration training data will be changed, we need to save corresponding labels
           Medels{z,1} = mode2;
           trainX1 = [misX;trainX1]; % combining the training data with miss classified samples
           trainY1 = [misY;trainY1];% combining the training lebels with miss classified labels
           for i=1:size (misX,2)
               b=corr(misX(:,i),misY); % Finding the error. however result id NaN because miss classified samples belong to single class.
               a=[a b];
           end
        m1=(sum(a))/size (misY,1);
        p=[p m1];
        m =[m abs(p(1,z)-p(1,z+1))];
        a=[];
%         l=l+1;
        falgA = double(isempty(misX));
        if  falgA 
            break
        end
  end

        % Finding Alpha
       alpha=FInd_Alpha(Trainingdata); % finding the alpha for training data of each iteration
       svml8 = [];
       svm_prede=[]
       %测试集合进行测试模型,得出预测的结果
      
       
       for i=1:size(Trainingdata)
           test1=testX*(Us{i}); % Coressponding U
           mod1=Medels{i}% Corresponding model
           fea=feature{i};% Coressponding Feature
           test1=test1(:,fea);
%             [svm_pred1,votes] = classRF_predict(test1,mod1)
           svm_pred1 = svmpredict(testY,test1,mod1);
           [ind]=find(svm_pred1==2); % Because Sgn function works with labels -1 and 1 so here I changed all my labels from 2 to -1 to make it possible(1,-1) before saving
           svm_pred1(ind,1)=-1;
           svm_pred1=svm_pred1*alpha(i); % Multiplication of alpha with corresponding prediction
           svm_prede=[svm_prede svm_pred1];
       end

       
      %% Prediction  预测精度计算
       [ind] = find(testY==2); % Changing labels having value 2 to -1 for calculation of accuracy
       testY(ind,1) = -1;      %把testY中为-2的改为-1.
       Final_predict=sum(svm_prede,2);
       Result=sign(Final_predict);    
       Final_Accuracy_with_Adaboosti = mean(Result == testY) * 100; % Final accuracy
%          Final_Accuracy_with_Adaboosti = ccalculateAccuracy(test_data,testY,Result);
       ensemble=[ensemble Result];
       FinalAcci = [FinalAcci Final_Accuracy_with_Adaboosti];
       
       %% 每一个子空间的评价指标
       Y_unique = unique(testY);
       %把每一个子空间的混淆矩阵都保存下来
       CMi{s} = zeros(size(Y_unique,1),size(Y_unique,1));
       for n = 1:2
           in = find(testY == Y_unique(n,1)); %找出实际标签为某一类标签的位置。
           Y_pred = Result(in,1);
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
       Spei(s,:) = TNi(s,:) / (FPi(s,:) + TNi(s,:)); %# 修正Spe 指标 
       

       G_meani(s,:) = sqrt(Reci(s,:) * Spei(s,:));
       F1_scorei(s,:) = (2 * Prei(s,:) * Reci(s,:)) / (Prei(s,:) + Reci(s,:))
       
       clear Trainingdata
 end
    
   load 'weight'%加载网格搜索法权重 
   m_w = size(weight,1);     %网格搜索方法找寻最佳权重
   for i = 1:m_w
       w1 = weight(i,:);
       %% 考虑这个是否加sign 
        P = sign(ensemble(:,1) * w1(1,1)+ ensemble(:,2) * w1(1,2) + ensemble(:,3) * w1(1,3));
       Final_Accuracy_with_ensebleL = mean(P == testY) * 100; % Final accuracy
       ACC_w1_all = [ACC_w1_all; Final_Accuracy_with_ensebleL];
   end
   %% 评价指标相关代码
   ACC_svml_tt = max(ACC_w1_all);
   [indx] = find(ACC_w1_all== max(max(ACC_w1_all)));
   best_weight = weight(indx,:);
   
   %best_P是最好的预测吧，利用预测计算混淆矩阵。
   best_Pred = sign(ensemble(:,1) * best_weight(1,1) + ensemble(:,2) * best_weight(1,2) + ensemble(:,3) * best_weight(1,3));
%      best_Pred = ensemble(:,1) * best_weight(1,1) + ensemble(:,2) * best_weight(1,2) + ensemble(:,3) * best_weight(1,3)
Final_Acc = calculateAccuracy(test_data,testY, best_Pred);
    
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
   
   TP = CM(1,1);
   FN = CM(1,2);
   FP = CM(2,1);
   TN = CM(2,2);
   %下面写评价指标
   ACC = (TP+TN) / (TP + TN + FP + FN);
   Pre = TP / (TP + FP);
   Rec = TP / (TP + FN);
   Spe = TN / (FP + TN);  %#修正SPE评价指标

   G_mean = sqrt(Rec * Spe);
   F1_score = (2 * Pre * Rec) / (Pre + Rec);
   %%
   fprintf('\nproposed with binary+svml Accuracy(train&test): %f\n', ACC_svml_tt);
   tr2=[tr2;ACC_svml_tt];   
   sakar_subject_Result.ACC = ACC;   
   sakar_subject_Result.ACCi = ACCi; 
   sakar_subject_Result.FinalAcci = FinalAcci
   sakar_subject_Result.Final_Acc = Final_Acc;
   sakar_subject_Result.Pre = Pre;
   sakar_subject_Result.Prei = Prei
   sakar_subject_Result.Rec = Rec; 
   sakar_subject_Result.Reci = Reci; 
   sakar_subject_Result.Spe = Spe; 
   sakar_subject_Result.Spei = Spei;
   sakar_subject_Result.G_mean = G_mean; 
   sakar_subject_Result.G_meani = G_meani; 
   sakar_subject_Result.F1_scorei = F1_scorei; 
   sakar_subject_Result.F1_score= F1_score; 
end
end




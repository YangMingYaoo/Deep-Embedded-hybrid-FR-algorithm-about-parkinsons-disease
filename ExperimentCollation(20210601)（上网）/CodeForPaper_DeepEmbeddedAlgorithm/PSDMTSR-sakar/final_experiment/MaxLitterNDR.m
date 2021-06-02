%% 
clc;clear ;warning('off');
tic
final_eval= {};
mean_final_eval = {};
iter = 10
for C = 1:iter
    [MaxLittlerProposed_H_experiment1_s2] = NDR()
    final_eval{C} = MaxLittlerProposed_H_experiment1_s2;
end
mean_final_eval.ACC = (final_eval{1}.ACC + final_eval{2}.ACC + final_eval{3}.ACC+final_eval{4}.ACC + final_eval{5}.ACC + final_eval{6}.ACC+final_eval{7}.ACC + final_eval{8}.ACC + final_eval{9}.ACC + final_eval{10}.ACC)/10
mean_final_eval.ACCi = (final_eval{1}.ACCi + final_eval{2}.ACCi + final_eval{3}.ACCi + final_eval{4}.ACCi + final_eval{5}.ACCi + final_eval{6}.ACCi + final_eval{7}.ACCi + final_eval{8}.ACCi + final_eval{9}.ACCi + final_eval{10}.ACCi)/10
mean_final_eval.Pre = (final_eval{1}.Pre + final_eval{2}.Pre + final_eval{3}.Pre + final_eval{4}.Pre + final_eval{5}.Pre + final_eval{6}.Pre + final_eval{7}.Pre + final_eval{8}.Pre + final_eval{9}.Pre + final_eval{10}.Pre)/10
mean_final_eval.Prei = (final_eval{1}.Prei + final_eval{2}.Prei + final_eval{3}.Prei + final_eval{4}.Prei + final_eval{5}.Prei + final_eval{6}.Prei + final_eval{7}.Prei + final_eval{8}.Prei + final_eval{9}.Prei + final_eval{10}.Prei)/10
mean_final_eval.Rec = (final_eval{1}.Rec + final_eval{2}.Rec + final_eval{3}.Rec + final_eval{4}.Rec + final_eval{5}.Rec + final_eval{6}.Rec + final_eval{7}.Rec + final_eval{8}.Rec + final_eval{9}.Rec + final_eval{10}.Rec)/10
mean_final_eval.Reci = (final_eval{1}.Reci + final_eval{2}.Reci + final_eval{3}.Reci + final_eval{4}.Reci + final_eval{5}.Reci + final_eval{6}.Reci + final_eval{7}.Reci + final_eval{8}.Reci + final_eval{9}.Reci + final_eval{10}.Reci)/10
mean_final_eval.Spe = (final_eval{1}.Spe + final_eval{2}.Spe + final_eval{3}.Spe + final_eval{4}.Spe + final_eval{5}.Spe + final_eval{6}.Spe + final_eval{7}.Spe + final_eval{8}.Spe + final_eval{9}.Spe + final_eval{10}.Spe)/10
mean_final_eval.Spei = (final_eval{1}.Spei + final_eval{2}.Spei + final_eval{3}.Spei + final_eval{4}.Spei + final_eval{5}.Spei + final_eval{6}.Spei + final_eval{7}.Spei + final_eval{8}.Spei + final_eval{9}.Spei + final_eval{10}.Spei)/10
mean_final_eval.G_mean = (final_eval{1}.G_mean + final_eval{2}.G_mean + final_eval{3}.G_mean + final_eval{4}.G_mean + final_eval{5}.G_mean + final_eval{6}.G_mean + final_eval{7}.G_mean + final_eval{8}.G_mean + final_eval{9}.G_mean + final_eval{10}.G_mean)/10
mean_final_eval.G_meani = (final_eval{1}.G_meani + final_eval{2}.G_meani + final_eval{3}.G_meani + final_eval{4}.G_meani + final_eval{5}.G_meani + final_eval{6}.G_meani + final_eval{7}.G_meani + final_eval{8}.G_meani + final_eval{9}.G_meani + final_eval{10}.G_meani)/10
mean_final_eval.F1_score = (final_eval{1}.F1_score + final_eval{2}.F1_score + final_eval{3}.F1_score + final_eval{4}.F1_score + final_eval{5}.F1_score+ final_eval{6}.F1_score + final_eval{7}.F1_score + final_eval{8}.F1_score + final_eval{9}.F1_score + final_eval{10}.F1_score)/10
mean_final_eval.F1_scorei = (final_eval{1}.F1_scorei + final_eval{2}.F1_scorei + final_eval{3}.F1_scorei + final_eval{4}.F1_scorei + final_eval{5}.F1_scorei + final_eval{6}.F1_scorei + final_eval{7}.F1_scorei + final_eval{8}.F1_scorei + final_eval{9}.F1_scorei + final_eval{10}.F1_scorei)/10
toc
%% 

function [NDR_Result] = NDR()
        % 记住RF时不返回AUC
        clear all;close all ;clc;
         %# 加载数据集 
         load('sakar_sample');%数据集读取
         
         ACC_w1_all=[];
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
         AUCi = {};
         svml3 = [];
         scores = [];
         ensemble = [];
         NDR_Result = [];
      
        for s = 1:3 
           %% 数据集处理部分
           % 拆分数据和标签
           trainX = traindataX{1,s}(:,2:end-1);
           trainY = traindataX{1,s}(:,end);
           validX = valid_data(:,2:end-1);  
           validY = valid_data(:,end);
           testX = test_data(:,2:end-1);
           testY = test_data(:,end);

           %# 样本标准化部分
           [trainX, mu, sigma] = featureCentralize(trainX);%%将样本标准化（服从N(0,1)分布）
           testX = bsxfun(@minus, testX, mu);
           testX = bsxfun(@rdivide, testX, sigma); 
           validX = bsxfun(@minus, validX, mu);
           validX = bsxfun(@rdivide, validX, sigma); 

      %% 使用所有变换后的训练集训练模型并使用变换后的测试集合预测数据
      
             % SVM 线性 
%        mode2 = svmtrain(trainY,trainX,'-s 0 -t 0 -b 1'); 
%        [svm_pred1,~,Scores] = svmpredict(testY,testX,mode2,'-b 1');
%        svml3 =  [svml3 calculateAccuracy(test_data,testY,svm_pred1)];


       % SVM 高斯 
       mode2 = svmtrain(trainY,trainX,'-s 0 -t 2 -b 1'); 
       [svm_pred1,~,Scores] = svmpredict(testY,testX,mode2,'-b 1');
       svml3 =  [svml3 calculateAccuracy(test_data,testY,svm_pred1)];
%       svml3 =  [svml3 mean(svm_pred1 == testY ) * 100];
       
%        mode2 = svmtrain(trainY,trainX,'-s 0 -t 2 -b 1'); 
%        [svm_pred1,~,Scores] = svmpredict(testY,testX,mode2,'-b 1');
%        svml3 =  [svml3 calculateAccuracy(test_data,testY,svm_pred1)];
%               
       
       % RF 
%      mode2 = classRF_train(trainX,trainY,'ntree',300)
%      [svm_pred1,votes] = classRF_predict(testX,mode2)
%    svml3 =  [svml3 calculateAccuracy(test_data,testY,svm_pred1)];

       %% 每一个子空间的模型评价指标
       % 因为后面程序按照标签为[-1,1]写的，所以这里把2变成-1.标签为2的是患者,把患者的标签变成-1
        [ind] = find(svm_pred1 == 2); % Because Sgn function works with labels -1 and 1 so here I changed all my labels from 2 to -1 to make it possible(1,-1) before saving
        svm_pred1(ind,1) = -1;
        % 把每一个子空间的预测保存下来
        ensemble = [ensemble svm_pred1];
        [ind] = find(testY==2); % Changing labels having value 2 to -1 for calculation of accuracy
        testY(ind,1) = -1;      %把testY中为-2的改为-1.
         
       Y_unique = unique(testY);
       
       %# 创建混淆矩阵
       CMi{s} = zeros(size(Y_unique,1),size(Y_unique,1));
       for n = 1:2
             in = find(testY == Y_unique(n,1)); %找出实际标签为某一类标签的位置。
             Y_pred =  svm_pred1(in,1);
             CMi{s}(n,1) = size(find(Y_pred == Y_unique(1,1)),1)%对于positive类.找到预测标签和真实标签相等的数量/对于negtivel类找到预测标签和真实标签不相等的数量
             CMi{s}(n,2) = size(find(Y_pred == Y_unique(2,1)),1)%对于positive类.找到预测标签和真实标签不相等的数量/对于negtivel类找到预测标签和真实标签相等的数量
       end
       % 把每一个子空间的混淆矩阵具体值记录下来
       TPi(s,:) = CMi{s}(1,1);
       FNi(s,:) = CMi{s}(1,2);
       FPi(s,:) = CMi{s}(2,1);
       TNi(s,:) = CMi{s}(2,2);
       
       ACCi(s,:) = (TPi(s,:) + TNi(s,:)) / (TPi(s,:) + TNi(s,:) + FPi(s,:) + FNi(s,:));
       Prei(s,:) = TPi(s,:) / (TPi(s,:) + FPi(s,:));
       Reci(s,:) = TPi(s,:) / (TPi(s,:) + FNi(s,:));
       Spei(s,:) = TNi(s,:) / (FPi(s,:) + TNi(s,:));  %#spe修正指标的修改
       
       G_meani(s,:) = sqrt(Reci(s,:) * Spei(s,:));
       F1_scorei(s,:) = 2 * ((Prei(s,:) * Reci(s,:)) / (Prei(s,:) + Reci(s,:)));
%        Evaluation_index_i(s,:) = [ACCi(s,:) Prei(s,:) Reci(s,:) Spei(s,:) G_meani(s,:) F1_scorei(s,:)]
    %% 在RF时候要注释掉//整理每一个子空间 Scores得分 
     % Scores 是每一个子空间的使用预测后的概率
%         scores = [scores Scores];    %scores里面存放的是每一个子空间中所有次训练得到的Scores. 也机存放的是一个子空间中的东西
     % 计算每一个子空间ROC曲线下方的面积AUC值
    %% 计算AUC值得程序段  ,本程序在RF时候要注释掉
%     让精度最好空间的AUC作为决策层的精度,也即最终的精度 
       Final_scores1 = [];
       Final_scores2 = [];
       Column_number_scores = size(scores,2); %求scores的列数
      for o = 1:Column_number_scores
          if mod(o,2)
             Final_scores1 = [Final_scores1 scores(:,o)] 
          else
             Final_scores2 = [Final_scores2 scores(:,o)] 
          end
      end 
     % 对多个模型得到分数取平均存放 得到每一个子空间的分数
     Mean_scores{s,1} = [(sum(Final_scores1,2)/(Column_number_scores/2)) (sum(Final_scores2,2)/(Column_number_scores/2))]; 
     %% RF时候隐藏下面这句代码 
%      AUCi{s} = plot_roc(Mean_scores{s,1}(:, 2),testY);  % Mean_scores{s,2}要注意输入的哪一列的概率值。
  end % 结束每一个子空间标志
        
  %% 下面部分使用网格权重搜索方法，集成三个子空间的预测
       % 加载权重矩阵
          load 'weight'%加载网格搜索法权重 
           m_w = size(weight,1);     %网格搜索方法找寻最佳权重
       for q = 1:m_w
            w1= weight(q,:);
            % 考虑这里是否添加sign
            P = sign(ensemble(:,1) * w1(1,1) + ensemble(:,2) * w1(1,2) + ensemble(:,3) * w1(1,3));
%              P  = ensemble(:,1) * w1(1,1) + ensemble(:,2) * w1(1,2) + ensemble(:,3) * w1(1,3);
             Final_Accuracy_with_ensebleL = mean(P == testY) * 100; % Final accuracy
            ACC_w1_all = [ACC_w1_all; Final_Accuracy_with_ensebleL];
       end
       %% 评价指标相关代码
       ACC_svml_tt = max(ACC_w1_all);
       indx = find(ACC_w1_all == max(max(ACC_w1_all)));
       best_weight = weight(indx(1,1),:);
        %best_P是最好的预测吧，利用预测计算混淆矩阵。
       best_Pred = sign(ensemble(:,1) * best_weight(1,1) + ensemble(:,2) * best_weight(1,2) + ensemble(:,3) * best_weight(1,3));
%        best_Pred = ensemble(:,1) * best_weight(1,1) + ensemble(:,2) * best_weight(1,2) + ensemble(:,3) * best_weight(1,3);
       
      
       % best_Pred 是预测标签，testY是真实标签。到了这一步之后，预测标签和真实标签都为1和-1，原本的2都转为-1了。
      Final_Acc = calculateAccuracy(test_data,testY, best_Pred);
      
      % # 创建决策层的混淆矩阵 。
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
   ACC = (TP+TN) / (TP + TN + FP + FN);
   Pre = TP / (TP + FP);
   Rec = TP / (TP + FN);
   Spe = TN / (FP + TN); %#spe修正指标的修改
   
   G_mean = sqrt(Rec * Spe);
   F1_score = 2 * ((Pre * Rec) / (Pre + Rec));
   %% RF时候要做修改
   %下面这句代码得到最高精度子空间的判别分数,用最高子空间的分数作为决策层空间的分数
%    Final_scores = Mean_scores{1} * best_weight(1,1) + Mean_scores{2} * best_weight(1,2) + Mean_scores{3} * best_weight(1,3)
%    AUC_final = plot_roc(Final_scores(:,2),testY);
%    Final_Evaluation_index = [ACC Pre Rec Spe G_mean  F1_score_Final  AUC_final ] 
   % RF时候换用这句话
   Final_Evaluation_index = [ACC Pre Rec Spe G_mean  F1_score  ] 
 
   %# 评价标准存储
   tr2 = [tr2;ACC_svml_tt]; 
   ACC_final_mean = mean(tr2); 
   NDR_Result.ACC_final_mean = ACC_final_mean;
   NDR_Result.ACC = ACC;   
   NDR_Result.ACCi = ACCi; 
   NDR_Result.acci_subject = svml3; 
   NDR_Result.acc_subject  = Final_Acc;
   NDR_Result.Pre = Pre;
   NDR_Result.Prei = Prei;
   NDR_Result.Rec = Rec; 
   NDR_Result.Reci = Reci;
   NDR_Result.Spe = Spe; 
   NDR_Result.Spei = Spei;
   NDR_Result.G_mean = G_mean; 
   NDR_Result.G_meani = G_meani; 
   NDR_Result.F1_score = F1_score; 
   NDR_Result.F1_scorei = F1_scorei; 
 
   NDR_Result.svml3 = svml3; 
end
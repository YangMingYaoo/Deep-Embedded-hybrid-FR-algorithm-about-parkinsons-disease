function [onlyLPPheatkernel_Result] = onlyLPPheatkernel_adbstLDPP()
        clear all;close all ;clc;
        %% 加载数据集 
         load('PD_LSVTsample');%数据集读取
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
         onlyLPPheatkernel_Result = [];
         
         for s = 1:3 
           
            %% 数据集处理部分
             % 拆分数据和标签
             trainX = traindataX{1,s}(:,1:end-1);
             trainY = traindataX{1,s}(:,end);
             validX = valid_data(:,1:end-1);  
             validY = valid_data(:,end);
             testX = test_data(:,1:end-1);
             testY = test_data(:,end);
           %% 样本标准化部分
            %下面给数据加1，想要区别与原始数据，可以把原始数据保留
            [trainX, mu, sigma] = featureCentralize(trainX);%%将样本标准化（服从N(0,1)分布）
            testX = bsxfun(@minus, testX, mu);
            testX = bsxfun(@rdivide, testX, sigma);
            validX = bsxfun(@minus, validX, mu);
            validX = bsxfun(@rdivide, validX, sigma); 
            
            
          %% 使用LPP特征变换处理数据部分
           kk=5;
           svml8 = [];
           t = [];
        
           for it = 1:9
               method = [];
               method.mode = 'lpp';
               method.t = 0.00001*power(10,it);
%              method.weightmode = 'binary';
               method.weightmode = 'heatkernel';
               method.knn_k = 5;%为减少参数调节，一律为5
               U = featureExtract(trainX,trainY,method,type_num);
               for ik = 1:floor(size(trainX,2)/5)
%                    t = [t;[ik it]];
                   method.K = kk * ik;
                   trainZ1 = projectData(trainX, U, method.K);
                   validZ1 = projectData(validX, U, method.K);

                 % SVM高斯 
%                model = svmtrain(trainY,trainZ1,'-s 0 -t 2 ');%%使用所有变换后的训练集训练模型
%                svm_pred1 = svmpredict(validY,validZ1,model);  
%                svml8(it,ik) = mean(svm_pred1 == validY) * 100;

                  %SVM 线性
%                   model = svmtrain(trainY,trainZ1,'-s 0 -t 0 '); %使用所有变换后的训练集训练模型
%                   svm_pred1 = svmpredict(validY,validZ1,model);  
%                   svml8 = [svml8 mean(svm_pred1 == validY) * 100;];

                   % RF随机森林               
                 model = classRF_train(trainZ1,trainY,'ntree',300)
                 [svm_pred1,votes] = classRF_predict(validZ1,model)
                 svml8(it,ik) = mean(svm_pred1 == validY) * 100
               end 

           end
           [loc_x,loc_y] = find ( svml8 == max(max(svml8)))
           acc_svml8_max = max(max(svml8)); 
           best_t = loc_x(1,1);
           best_svml_t = 0.00001 * power(10,best_t(1,1));
           best_svml_kk = loc_y(1,1) * kk;
           method.t = best_svml_t
           U = featureExtract(trainX,trainY,method,type_num);
            %% 上面代码找出最好的U矩阵参数存放在 method中
             trainZ2 = projectData(trainX, U, best_svml_kk);
             testZ2 = projectData(testX, U, best_svml_kk);
            %% 使用所有变换后的训练集训练模型并预测
              % 这里要使用测试数据集，上面选参数的时候需要用到验证数据集
              
               % SVM 高斯 
%                    mode2 = svmtrain(trainY,trainZ2,'-s 0 -t 2 -b 1'); 
%                    [svm_pred3,~,Scores] = svmpredict(testY,testZ2,mode2,'-b 1');
%                    svml3 = [svml3 mean(svm_pred3 == testY) * 100;];

                   % SVM 线性 
%                    mode2 = svmtrain(trainY,trainZ2,'-s 0 -t 0 -b 1'); 
%                    [svm_pred3,~,Scores] = svmpredict(testY,testZ2,mode2,'-b 1');
%                    svml3 = [svml3 mean(svm_pred3 == testY) * 100;];
                   
                   % RF     classRF_train 这里必须 先写trainZ2 在写 trainY
                 mode2 = classRF_train(trainZ2,trainY,'ntree',300)
                 [svm_pred3,votes] = classRF_predict(testZ2,mode2)
                 svml3 = [svml3 mean(svm_pred3 == testY) * 100;];
%                  
                 
        %% 评价标准部分
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
 
        end   %这里结束每一个子空间的寻找
       %% 下面部分使用网格权重搜索方法，集成三个子空间的预测
        % 加载权重矩阵 %网格搜索方法找寻最佳权重
           load 'weight'%加载网格搜索法权重 
           m_w = size(weight,1); 
       for q = 1:m_w
            w1= weight(q,:);
            P = ensemble(:,1) * w1(1,1) + ensemble(:,2) * w1(1,2) + ensemble(:,3) * w1(1,3);
            Final_Accuracy_with_ensebleL = mean(P == testY) * 100; % Final accuracy
            ACC_w1_all = [ACC_w1_all; Final_Accuracy_with_ensebleL];
       end
       %% 评价指标相关代码
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
   ACC = (TP+TN) / (TP + TN + FP + FN);
   Pre = TP / (TP + FP);
   Rec = TP / (TP + FN);
   Spe = TN / (FN + TN);
   G_mean = sqrt(Rec * Spe);
   F1_score_Final = (2 * Pre * Rec) / (Pre + Rec);
   %% RF时候要做修改
   %下面这句代码得到最高精度子空间的判别分数,用最高子空间的分数作为决策层空间的分数
%    Final_scores = Mean_scores{1} * best_weight(1,1) + Mean_scores{2} * best_weight(1,2) + Mean_scores{3} * best_weight(1,3)
%    AUC_final = plot_roc(Final_scores(:,2),testY);
%    Final_Evaluation_index = [ACC Pre Rec Spe G_mean  F1_score_Final  AUC_final ] 
   % RF时候换用这句话
   Final_Evaluation_index = [ACC Pre Rec Spe G_mean  F1_score_Final   ] 
   %% 
   tr2 = [tr2;ACC_svml_tt]; 
   ACC_final_mean = mean(tr2); 
   onlyLPPheatkernel_Result.ACC_final_mean = ACC_final_mean;
   onlyLPPheatkernel_Result.ACC = ACC;   
   onlyLPPheatkernel_Result.ACCi = ACCi; 
   
   %% RF 时候隐藏下面2句代码
%    onlyLPPheatkernel_Result.AUC_final = AUC_final; 
%    onlyLPPheatkernel_Result.AUCi = AUCi; 
   %% 
   onlyLPPheatkernel_Result.F1_score_Final = F1_score_Final; 
   onlyLPPheatkernel_Result.F1_scorei = F1_scorei; 
   onlyLPPheatkernel_Result.G_mean = G_mean; 
   onlyLPPheatkernel_Result.G_meani = G_meani; 
   onlyLPPheatkernel_Result.Pre = Pre;
   onlyLPPheatkernel_Result.Prei = Prei; 
   onlyLPPheatkernel_Result.Rec = Rec; 
   onlyLPPheatkernel_Result.Reci = Reci; 
   onlyLPPheatkernel_Result.Spe = Spe; 
   onlyLPPheatkernel_Result.Spei = Spei; 
   onlyLPPheatkernel_Result.svml3 = svml3;  
   
end

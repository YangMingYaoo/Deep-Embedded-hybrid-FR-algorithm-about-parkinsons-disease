function [] = sakar_ldpp_relief()
clear ; close all; clc;warning('off');
load Sakar_PDsample1;%数据集读取
load('weight.mat');
% load('train_x.mat');
tr2 = []
tic
% tr_x=[];
% train_data_all = trainX;
% train_label_all = trainY;
% test_data = testX;
% type_num = 2;
for iter=1:1
%     [train_x]=RandomSampling()
    

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
        method.weightmode = 'binary';
        method.knn_k = 5;
        svml8 = [];
        for i = 1:3
            trainX=train_data_all(:,1:end);
            testX=test_data(:,1:end);
            trainY=train_label_all;
          %% 给样本添加噪声
            train_cqX=train_x{iter,i}(:,1:end-1);
            train_cqY=train_x{iter,i}(:,end);
            [train_cqX, mu, sigma] = featureCentralize(train_cqX);
            testX = bsxfun(@minus, testX, mu);
            testX = bsxfun(@rdivide, testX, sigma);%%将测试样本标准化
            trainX=bsxfun(@minus, trainX, mu);
            trainX=bsxfun(@rdivide, trainX, sigma);%%将所有训练样本标准化
            U = featureExtract2(train_cqX,train_cqY,method,type_num);
            U1_all{1,i}=U;
            for ik=1:floor(size(trainX,2)/5)
                method.K = kk*ik;
                mukgamma=[mukgamma;[imu ik igamma]];    
                trainZ1=projectData(trainX, U, method.K);
                testZ = projectData(testX, U, method.K);
                
                
                
                % SVM
%                 model = svmtrain(trainY,trainZ1,'-s 0 -t 2');%%???????????????
%                 svm_pred = svmpredict(testY,testZ,model);
                %% RF
%         model = classRF_train(trainZ1,trainY,'ntree',300);
%         [svm_pred,votes] = classRF_predict(testZ,model);
%         svml8(i,ik)=mean(svm_pred == testY) * 100;
%  ELM         
% [~, ~, ~, TestingAccuracy,svm_pred] = ELM([trainY trainZ1], [testY testZ], 1, 5000, 'sig', 10^2);
%                 svml8(i,ik)=mean(svm_pred == testY) * 100;
                
%                 svml8(i,ik)=TestingAccuracy * 100;
%                 
                model =svmtrain(trainY,trainZ1,'-s 0 -t 0');%%使用所有变换后的训练集训练模型
                svm_pred = svmpredict(testY,testZ,model);
                svml8(i,ik)=mean(svm_pred == testY) * 100;
            end
        end
%         U1=(U1_all{1,1}+U1_all{1,2}+U1_all{1,3})/3;
        [loc_x,loc_y]=find(svml8==max(max(svml8)));%找到最大值的位置
%         [loc_x,loc_y]=max(mean(svml8));%找到最大值的位置
        if max(max(svml8))>mean_svml8_max
            mean_svml8_max=max(max(svml8));
            U_svml_best=U1_all;
            best_svml_kk = kk*loc_y(1,1);
            best_svml_mu=0.00001*power(10,imu);%取[0.0001,0.001,...,1000,10000]
            best_svml_gamma=0.00001*power(10,igamma);
        end
    end
end

m_w=size(weight,1);
ACC_w1_all=[];
    testX=test_data(:,1:end);
    trainX=train_data_all(:,1:end);    
    [trainX, mu, sigma] = featureCentralize(trainX);
    testX = bsxfun(@minus, testX, mu);
    testX = bsxfun(@rdivide, testX, sigma);%%将test数据标准化
    kk=5;
svml7=[];
ACC_w=[];
tr=[];
for i=1:m_w
    w1=weight(i,:);
    U_mean=U_svml_best{1,1}*w1(1,1)+U_svml_best{1,2}*w1(1,2)+U_svml_best{1,3}*w1(1,3);
    train =projectData(trainX, U_mean, best_svml_kk);
    testZ = projectData(testX, U_mean, best_svml_kk);
    
% [svml7]=Reliefme(train,trainY,testZ,testY)
[fea] = relieff(train,trainY, 5);%5 means the nunber of the nerbor
for p=1:floor(size(train,2));
%     K = kk*ik;
    trainZ1=train(:,fea(:,1:p));
    testZ1 = testZ(:,fea(:,1:p));
%    SVM 
    model = svmtrain(trainY,trainZ1,'-s 0 -t 0');%%
    svm_pred = svmpredict(testY,testZ1,model);
    svml7 = [svml7 calculateAccuracy(testX,testY,svm_pred)];
%     a(ik)=K;
%     ACC_w=[ACC_w svml7];%%得到本文算法的分类准确率
%     tr=[tr;svml7];
% end

      % RF
%
%         model = classRF_train(trainZ1,trainY,'ntree',300);
%         [svm_pred,votes] = classRF_predict(testZ1,model);
%         svml7=mean(svm_pred == testY) * 100;
%         ACC_w=[ACC_w svml7]
      % ELM

%     [~, ~, ~, TestingAccuracy,svm_pred] = ELM([trainY trainZ1], [testY testZ1], 1, 5000, 'sig', 10^2);
%           svml7=mean(svm_pred == testY) * 100;
%     ACC_w=[ACC_w TestingAccuracy];%%得到本文算法的分类准确率
%     tr=[tr;svml7];

end
ACC_w1_all=[ACC_w1_all; svml7];
% tr=[tr ACC_w1_all];
svml7=[];
end
[ACC_svml_tt,indx]=max(max(ACC_w1_all));
% weight_svml1_fitmax=weight(indx,:);
fprintf('\nproposed with binary+svml Accuracy(train&test): %f\n', ACC_svml_tt);
% save('ldpp_u8.mat','best_svml_mu','best_svml_kk','best_svml_gamma','ACC_svml_tt','weight_svml1_fitmax');

 tr2=[tr2;ACC_svml_tt];
 tr_x=[tr_x;train_x];
 
end
me = sum(tr2)/iter;
maxxx = max(tr2);
minn = min(tr2);
toc
end

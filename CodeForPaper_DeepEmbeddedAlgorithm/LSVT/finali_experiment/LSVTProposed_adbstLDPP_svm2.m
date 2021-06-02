function [f,U1,mode2,indx,missed_samples,missed_labels] = LSVTProposed_adbstLDPP_svm2(trainx,trainy,validx,validy,type_num)

kk=5;
mukgamma=[];
mean_svml8_max=0;
for igamma=1:9
    for imu=1:9
        method = [];
        method.mode = 'ldpp_u';
        method.mu = 0.00001*power(10,imu);
        method.gamma = 0.00001*power(10,igamma);
        method.M = 200;
        method.labda2 = 0.001;   %ȡ[0.0001,0.001,...,1000,10000]
        method.ratio_b = 0.9;    %��ʱ���Ǻ�������������ĺ���
        method.ratio_w = 0.9;
        method.weightmode = 'binary';
%         method.weightmode = 'heatkernel'; %��������Ը���ģʽ
        method.knn_k = 5;
        U = featureExtract2(trainx,trainy,method,type_num); %ʹ��֮ǰ�ķ�����ȡ����ӳ�����U
           for ik = 1:floor(size(trainx,2)/5)
                method.K = kk*ik;
                mukgamma = [mukgamma;[imu ik igamma]];    
                trainZ1 = projectData(trainx, U, method.K);
                testZ = projectData(validx, U, method.K);
                
%                 % SVM��˹ 
                model = svmtrain(trainy,trainZ1,'-s 0 -t 2');%%ʹ�����б任���ѵ����ѵ��ģ��
                svm_pred = svmpredict(validy,testZ,model);  % �����testZ����valid
                svml8(ik)= mean(svm_pred == validy) * 100;

                %SVM ����
%                 model = svmtrain(trainy,trainZ1,'-s 0 -t 0'); %%ʹ�����б任���ѵ����ѵ��ģ��
%                 svm_pred = svmpredict(validy,testZ,model);  % �����testZ����valid
%                 svml8(ik)= mean(svm_pred == validy) * 100;
                
                % RF���ɭ��               
%                 model = classRF_train(trainZ1,trainy,'ntree',300)
%                 [svm_pred,votes] = classRF_predict(testZ,model)
%                 svml8(ik) = mean(svm_pred == validy) * 100
                
                     %  ELM         
%                  [~, ~, ~, TestingAccuracy,svm_pred] = ELM([trainy trainZ1], [validy testZ], 1, 5000, 'sig', 10^2);
           end
           
           
           %%%%% ���Բ���%%%% 
%             method.K = 310
%            trainZ1 = projectData(trainx, U, method.K);
%                 testZ = projectData(validx, U, method.K);
%                 
% %                 % SVM��˹ 
%                 model = svmtrain(trainy,trainZ1,'-s 0 -t 2');%%ʹ�����б任���ѵ����ѵ��ģ��
%                 svm_pred = svmpredict(validy,testZ,model);  % �����testZ����valid
%                 svml8(ik)= mean(svm_pred == validy) * 100;

           
           
           
           %%%%%���Բ���%%%%%
           
           
           
           
            
               [acc_svml_max,indx2] = max(svml8);
               Accuracy(igamma,imu) = acc_svml_max;
               best_svml_kk = kk * indx2; 
               bestK(igamma,imu) = best_svml_kk;
    end
end
        [loc_x,loc_y] = find(Accuracy==max(max(Accuracy)));%�ҵ����ֵ��λ��
        mean_svml8_max = max(max(Accuracy));
%       U_svml_best=U1_all;
        best_svml_kk = bestK(loc_x(1),loc_y(1));
        method.mu = 0.00001 * power(10,loc_y(1));%ȡ[0.0001,0.001,...,1000,10000]
        method.gamma = 0.00001 * power(10,loc_x(1));
        
       %% ��������ҳ���õ�U������������ method��ͨ
        U = featureExtract2(trainx,trainy,method,type_num);  %ʹ���ҵ�����õ�U�������method�ó����Ų�����Ӧ��U����
        U1 = U(:,1:best_svml_kk);
        trainZ1 = trainx * U(:,1:best_svml_kk);  %ʹ����õ�U������б任
        testZ = validx * U(:,1:best_svml_kk);
%    U1 = U 
%             trainZ1 = trainx * U
%             testZ = validx * U
        %% ʹ��relief����ѡ���㷨
         [fea] = relieff(trainZ1,trainy, 5);
%          kk = 5;
         svml2 = [];
         % ������� ��������������������ѵ����Ԥ�⾫��  
        for ik=1:floor(size(trainZ1,2)/5) 
            K = kk * ik;  
            trainZ = trainZ1(:,fea(:,1:K));
            test = testZ(:,fea(:,1:K));
            
            % SVM��˹ 
              
             model = svmtrain(trainy,trainZ,'-s 0 -t 2');%%ʹ�����б任���ѵ����ѵ��ģ��
             svm_pred = svmpredict(validy,test,model);
             svml2 = [svml2 mean(svm_pred == validy) * 100;]; 
             

              % SVM ����
%              model = svmtrain(trainy,trainZ,'-s 0 -t 0');%%ʹ�����б任���ѵ����ѵ��ģ��
%              svm_pred = svmpredict(validy,test,model);
%              svml2 = [svml2 mean(svm_pred == validy) * 100;];
%                 
                % RF���ɭ��
%               model = classRF_train(trainZ,trainy,'ntree',300)
%              [svm_pred,votes] = classRF_predict(test,model)
%               svml2 = mean(svm_pred == validy) * 100;
%                 
                

           
        end
       % ������� ѡ������ŵ����� ����˼·���Ƚ���������ȡ���ڽ�������ѡ��û��������������������������
        [acc_svml_max,indx2] = max(svml2);
        best_svml_kk = kk * indx2;
        
      
       %% ��ѡ������������������ Ӧ�õ�LDPP����ӳ�������ݿ�
        f=fea(:,1:best_svml_kk);
        train = trainZ1(:,f);
        test = testZ(:,f);
               
      %% ʹ�����б任���ѵ����ѵ��ģ�Ͳ�Ԥ��
      
      % SVM ��˹ 
       mode2 = svmtrain(trainy,train,'-s 0 -t 2'); 
       svm_pred = svmpredict(validy,test,mode2);

       % SVM ���� 
%        mode2 = svmtrain(trainy,train,'-s 0 -t 0'); 
%        svm_pred = svmpredict(validy,test,mode2);
     
       % RF 
%      mode2 = classRF_train(train,trainy,'ntree',300)
%      [svm_pred,votes] = classRF_predict(test,mode2)
       

    
      %% �ҳ����ֵ������������������ص�������
       [indx,val]=find(0==(svm_pred==validy));    %Finding miss classified samples
       missed_samples = validx(indx,:);
       missed_labels = validy(indx,:);
       
       
end
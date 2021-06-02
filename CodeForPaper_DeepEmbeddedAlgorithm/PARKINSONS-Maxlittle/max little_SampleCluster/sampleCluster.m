function Y = sampleCluster(X,N)

[Idx1,C1] = kmeans(X(:,2:end-1),N);  % 如果有时候第一列是受试者标签，要考虑下
ctLb = []; %所有聚类中心的标签
B_H = [];   % 用来存储受试者的编号

%K均值聚类成簇，在对应的原始样本中，找出归为同一簇的样本，把该簇样本的

for i = 1:N  
    
    t1 = find(Idx1 == i);
    t2 = X (t1, :);%第i聚类的所有行样本
    t3 = t2 (:, end);%第i聚类的所有样本的标签
    t4 = mean(t3); 
    ctLb = [ctLb; t4];  
%     t5 = t2(:, 1);%第i聚类的所有样本的受试者编号
%     subject_index = [subject_index; t5]; % 受试者对应的编号

    
    t6 = mean(X(t1, 1)); %# 受试者编号
    B_H = [B_H;t6 ]  
   
end

Y = [B_H C1 ctLb];  %B_H受试者编号 C1,质心或者原样本 ctLb 标签、

end

% A = size(unique(Idx1))
%%分类情况
% for i = 1:N
%     t1 = find(idx==i);
%     t2 = X(t1, :);%第i聚类的所有样本
%     t3 = t2(:, end);%第i聚类的所有样本的标签
%     t4 = t2(:, 1:(end-1));
%     t5 = unique(t3);
%     if(length(t3)==length(t5))
%         t6 = knnclassify(ct(i, :), t4, t3);
%         ctLb = [ctLb; t6];
%     else
%         ctLb = [ctLb; mode(t3)];
%     end
% end
% 
% Y = [C1, ctLb];


%% 
% len=length(X);
% for i=1:N
%     c_0=[];
%     data_age_all=0;
%    a=0;
%     for j=1:len
%         if X(j,8)==i;
%             c0=j;
%             c_0=[c_0;c0];
%             a=a+1;
%             data_age=X(j,7);
%             data_age_all=data_age_all+data_age;
%         end
%     end
%     data_age_new=data_age_all/a;
%     m=length(c_0);
%     for k=1:m
%         X(c_0(k),7)=data_age_new;
%     end
% end
% data_arr=X;
% end

    
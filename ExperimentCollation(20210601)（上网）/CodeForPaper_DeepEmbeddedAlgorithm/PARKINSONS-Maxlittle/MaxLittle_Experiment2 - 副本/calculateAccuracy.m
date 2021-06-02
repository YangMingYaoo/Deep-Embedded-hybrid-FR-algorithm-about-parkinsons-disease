function acc=calculateAccuracy(data,dataY,pred)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % data为测试集数据，第一列为人物样本序号
% % dataY为测试集标签
% % pred为预测的标签
data_in = [data(:,1) dataY pred];
person_xh = unique(data_in(:,1));
n_p=size(person_xh,1);
count=0;

for i=1:n_p
    data_per=[];
    in=find(data_in(:,1)==person_xh(i,1));
    data_per=data_in(in,:);
    if(size(find(data_per(:,2)==data_per(:,3)),1)/size(in,1)>0.5)
        count=count+1;
    end
end
acc=count/n_p*100;
end
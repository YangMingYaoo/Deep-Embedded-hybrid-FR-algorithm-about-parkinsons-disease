function in_pvalue=sub_pvalue_selection(train_data_all,train_label_all)
Y=unique(train_label_all);
class=cell(1,size(Y,1));
for i=1:size(Y,1)
    in=find(train_label_all==Y(i,1));
    class{1,i}=train_data_all(in,:);
    m(1,i)=size(in,1);
end
m_min=min(m);
[h,p,ci]=ttest(class{1,1}(1:m_min,:),class{1,2}(1:m_min,:));
[p_order,in_pvalue]=sort(p);
end

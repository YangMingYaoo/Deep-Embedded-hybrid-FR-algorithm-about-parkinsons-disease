% 产生数据
clc;clear all;close all;
load 绘图1
Z = Accuracy;
figure
% subplot(1,2,1)
b = bar3(Z);
colorbar
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
xlabel('imu');ylabel('igamma');zlabel('Accuracy(%)')

set(gca,'YTickLabel',{'0.0001','0.001','0.01','0.1','1','10','100','1000','10000'});
set(gca,'XTickLabel',{'0.0001','0.001','0.01','0.1','1','10','100','1000','10000'});


subplot(1,2,2)
b = bar3(Z);
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'flat';
end


% %% 
result = Accuracy;
h = bar3(result);
for i = 1:length(h)
   c     = get(h(i), 'CData');
   color = repelem(repmat(colorInd(:, i), 1, 4), 6, 1);
   set(h(i), 'CData', color);
end
% for n=1:numel(h)
%     cdata=get(h(n),'zdata');
%     set(h(n),'cdata',cdata,'facecolor','interp')
% end
xlabel('μ');ylabel('γ');zlabel('Accuracy(%)')

set(gca,'YTickLabel',{'0.0001','0.001','0.01','0.1','1','10','100','1000','10000'});
set(gca,'XTickLabel',{'0.0001','0.001','0.01','0.1','1','10','100','1000','10000'});
legend('68','75','78','82','86','89','92','96');

%%%


%%%% 
clc;clear all; close all;
M=rand(30,20);
figure
subplot(1,2,1)
h=bar3(M);
for n=1:numel(h)
    cdata=get(h(n),'zdata');
    set(h(n),'cdata',cdata,'facecolor','interp')
end

subplot(1,2,2)
h=bar3(M);
for n=1:numel(h)
    cdata=get(h(n),'zdata');
    cdata=repmat(max(cdata,[],2),1,4);
    set(h(n),'cdata',cdata,'facecolor','flat')
end 
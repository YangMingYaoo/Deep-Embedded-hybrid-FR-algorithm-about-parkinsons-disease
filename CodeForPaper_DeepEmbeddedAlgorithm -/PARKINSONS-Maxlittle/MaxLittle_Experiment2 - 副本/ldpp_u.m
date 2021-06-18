function [U, S]=ldpp_u(X,Y,A,mu,gamma,type_num,ratio_b,ratio_w,M,yita)
[N, D] = size(X);
U = zeros(D);
S = zeros(D);
SB=zeros(D); 
m = mean(X);
centroids_b = zeros(type_num,D);
lengthIndex_b = zeros(1,type_num);
[part_X,part_Y,idx] = pick_neighbor2(X,Y,m,round(N*ratio_b));

for i = 1:type_num
    index_b = find(part_Y == i);
        if isempty(index_b)
            index_b = find(Y == i,1);
        end
        lengthIndex_b(i) = size(index_b,1);
        for j = 1:lengthIndex_b(i)
            centroids_b(i,:) = part_X(index_b(j),:)+centroids_b(i,:);
        end
    centroids_b(i,:) = centroids_b(i,:)/lengthIndex_b(i);
end
for i=1:type_num
    SB=SB+lengthIndex_b(i)*(centroids_b(i,:)-m)'*(centroids_b(i,:)-m);
end

SW=zeros(D); 
centroids_w = zeros(type_num,D);
for i=1:type_num
    index_w = find(Y == i);
    N_index = size(index_w,1);
    mu_X = mean(X(index_w,:));
    [local_X,local_Y] = pickNeighbor(X(index_w,:),Y(index_w),mu_X,round(N_index*ratio_w));
    lengthIndex_w = size(local_Y,1);
    for j = 1:lengthIndex_w
        centroids_w(i,:) = local_X(j,:)+centroids_w(i,:);
    end
    centroids_w(i,:) = centroids_w(i,:)/lengthIndex_w;
    for j = 1:lengthIndex_w
        SW=SW+(local_X(j,:)-centroids_w(i,:))'*(local_X(j,:)-centroids_w(i,:));
    end
end

Dist_m = EuDist2(centroids_b,centroids_b,0);
Dist_m = Dist_m + 10000*eye(size(centroids_b,1));
[R,~]=find(Dist_m==min(Dist_m(:)));

D = diag(sum(A,2));
L = D - A;

% matrix=pinv(mu*SB-gamma*(X'*L*X))*(SW+S_yita);
% matrix=pinv(mu*SB-gamma*(X'*L*X))*(SW+repmat(S_labda,size(SW,1),1));
matrix=pinv(mu*SB-gamma*(X'*L*X))*SW;
[U,S,V] = svd(matrix);
[val,ind] = sort(diag(S));%%求解特征值，并将其排序
U = U(:,ind);
S = S(ind,:);
end

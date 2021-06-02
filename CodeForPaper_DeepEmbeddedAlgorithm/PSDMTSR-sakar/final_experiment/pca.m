function [U, S] = pca(X)

[m, n] = size(X);

U = zeros(n);
S = zeros(n);

Sigma = X'*X;
[U,S,V] = svd(Sigma);

end
%% 程序介绍
% 程序输入一个输入矩阵,即需要经过pca变换的矩阵,
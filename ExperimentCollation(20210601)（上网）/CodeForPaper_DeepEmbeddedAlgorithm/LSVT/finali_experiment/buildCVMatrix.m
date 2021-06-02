%该函数为建立交叉矩阵函数
function [trnM, tstM] = buildCVMatrix(N, nfold)
blockSize = floor(N/nfold);
trnM = zeros(N, nfold);
tstM = zeros(N, nfold);

for i = 1:(nfold-1)
    index = repmat(false, N, 1);%把指引数组归0,设置1-9折交叉验证
    index(((i-1)*blockSize+1):(i*blockSize)) = true;
    tstM(index, i) = true;
    trnM(~index, i) = true;
end
index = repmat(false, N, 1);   %再次把指引数组归0 ,设置第10折交叉验证
index( ((nfold-1)*blockSize+1):N ) = true;
tstM(index, nfold) = true;
trnM(~index, nfold) = true;
end
% 这个函数是设置交叉验证矩阵,比如说10折交叉验证,在每一折中设置一部分为测试集合,其余的设为训练集. 
% 如果总数据样本的行数正好是要设置的测试样本数据行书的倍数,那么在一个for循环中可以完成1-10折,
% 比如:该函数之所以没有在第一个for循环中完成1-10折,原因是在某一次调用函数时候,总样本为42个,测试样本数据为4个
% 不是倍数关系
function stump = initStump(dim)
stump.dim = dim;
stump.error = 1e6;
stump.threshold = [];
stump.less = 1;
stump.more = -1; %这是不是两类 1和-1
end

%该函数初始化弱分类器
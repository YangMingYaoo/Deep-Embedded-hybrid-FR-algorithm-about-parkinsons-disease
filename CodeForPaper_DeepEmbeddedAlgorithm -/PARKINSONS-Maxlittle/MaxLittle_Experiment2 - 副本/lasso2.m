function [B,stats] = lasso2(X,Y,varargin)
%LASSO Perform lasso or elastic net regularization for linear regression.
%   [B,STATS] = lasso(X,Y,...) Performs L1-constrained linear least  
%   squares fits (lasso) or L1- and L2-constrained fits (elastic net)
%   relating the predictors in X to the responses in Y. The default is a
%   lasso fit, or constraint on the L1-norm of the coefficients B.
%
%   Positional parameters:
%
%     X                A numeric matrix (dimension, say, NxP)
%     Y                A numeric vector of length N
%   
%   Optional input parameters:  
%
%     'Weights'        Observation weights.  Must be a vector of non-negative
%                      values, of the same length as columns of X.  At least
%                      two values must be positive. (default ones(N,1) or 
%                      equivalently (1/N)*ones(N,1)).
%     'Alpha'          Elastic net mixing value, or the relative balance
%                      between L2 and L1 penalty (default 1, range (0,1]).
%                      Alpha=1 ==> lasso, otherwise elastic net.
%                      Alpha near zero ==> nearly ridge regression.
%     'NumLambda'      The number of lambda values to use, if the parameter
%                      'Lambda' is not supplied (default 100).  Ignored
%                      if 'Lambda' is supplied.  LASSO may return fewer
%                      fits than specified by 'NumLambda' if the residual
%                      error of the fits drops below a threshold percentage 
%                      of the variance of Y.
%     'LambdaRatio'    Ratio between the minimum value and maximum value of
%                      lambda to generate, if the  parameter "Lambda" is not 
%                      supplied.  Legal range is [0,1). Default is 0.0001.
%                      If 'LambdaRatio' is zero, LASSO will generate its
%                      default sequence of lambda values but replace the
%                      smallest value in this sequence with the value zero.
%                      'LambdaRatio' is ignored if 'Lambda' is supplied.
%     'Lambda'         Lambda values. Will be returned in return argument
%                      STATS in ascending order. The default is to have LASSO
%                      generate a sequence of lambda values, based on 'NumLambda'
%                      and 'LambdaRatio'. LASSO will generate a sequence, based
%                      on the values in X and Y, such that the largest LAMBDA                 
%                      value is just sufficient to produce all zero coefficients B.
%                      You may supply a vector of real, non-negative values of 
%                      lambda for LASSO to use, in place of its default sequence.
%                      If you supply a value for 'Lambda', 'NumLambda' and 
%                      'LambdaRatio' are ignored.
%     'DFmax'          Maximum number of non-zero coefficients in the model.
%                      Can be useful with large numbers of predictors.
%                      Results only for lambda values that satisfy this
%                      degree of sparseness will be returned. Default is
%                      to not limit the number of non-zero coefficients.
%     'Standardize'    Whether to scale X prior to fitting the model
%                      sequence. This affects whether the regularization is
%                      applied to the coefficients on the standardized
%                      scale or the original scale. The results are always
%                      presented on the original data scale. Default is
%                      TRUE, do scale X.
%                      Note: X and Y are always centered.
%     'RelTol'         Convergence threshold for coordinate descent algorithm.
%                      The coordinate descent iterations will terminate
%                      when the relative change in the size of the
%                      estimated coefficients B drops below this threshold.
%                      Default: 1e-4. Legal range is (0,1).
%     'CV'             If present, indicates the method used to compute MSE.
%                      When 'CV' is a positive integer K, LASSO uses K-fold
%                      cross-validation.  Set 'CV' to a cross-validation 
%                      partition, created using CVPARTITION, to use other
%                      forms of cross-validation. You cannot use a
%                      'Leaveout' partition with LASSO.                
%                      When 'CV' is 'resubstitution', LASSO uses X and Y 
%                      both to fit the model and to estimate the mean 
%                      squared errors, without cross-validation.  
%                      The default is 'resubstitution'.
%     'MCReps'         A positive integer indicating the number of Monte-Carlo
%                      repetitions for cross-validation.  The default value is 1.
%                      If 'CV' is 'resubstitution' or a cvpartition of type
%                      'resubstitution', 'MCReps' must be 1.  If 'CV' is a
%                      cvpartition of type 'holdout', then 'MCReps' must be
%                      greater than one.
%     'PredictorNames' A cell array of names for the predictor variables,
%                      in the order in which they appear in X. 
%                      Default: {}
%     'Options'        A structure that contains options specifying whether to
%                      conduct cross-validation evaluations in parallel, and
%                      options specifying how to use random numbers when computing
%                      cross validation partitions. This argument can be created
%                      by a call to STATSET. CROSSVAL uses the following fields:
%                        'UseParallel'
%                        'UseSubstreams'
%                        'Streams'
%                      For information on these fields see PARALLELSTATS.
%                      NOTE: If supplied, 'Streams' must be of length one.
%   
%   Return values:
%     B                The fitted coefficients for each model. 
%                      B will have dimension PxL, where 
%                      P = size(X,2) is the number of predictors, and
%                      L = length(lambda).
%     STATS            STATS is a struct that contains information about the
%                      sequence of model fits corresponding to the columns
%                      of B. STATS contains the following fields:
%
%       'Intercept'    The intercept term for each model. Dimension 1xL.
%       'Lambda'       The sequence of lambda penalties used, in ascending order. 
%                      Dimension 1xL.
%       'Alpha'        The elastic net mixing value that was used.
%       'DF'           The number of nonzero coefficients in B for each
%                      value of lambda. Dimension 1xL.
%       'MSE'          The mean squared error of the fitted model for each
%                      value of lambda. If cross-validation was performed,
%                      the values for 'MSE' represent Mean Prediction
%                      Squared Error for each value of lambda, as calculated 
%                      by cross-validation. Otherwise, 'MSE' is the mean
%                      sum of squared residuals obtained from the model
%                      with B and STATS.Intercept.
%
%     If cross-validation was performed, STATS also includes the following
%     fields:
%
%       'SE'           The standard error of MSE for each lambda, as
%                      calculated during cross-validation. Dimension 1xL.
%       'LambdaMinMSE' The lambda value with minimum MSE. Scalar.
%       'Lambda1SE'    The largest lambda such that MSE is within 
%                      one standard error of the minimum. Scalar.
%       'IndexMinMSE'  The index of Lambda with value LambdaMinMSE.
%       'Index1SE'     The index of Lambda with value Lambda1SE.
%
%     Examples:
%
%        % (1) Run the lasso on data obtained from the 1985 Auto Imports Database 
%        % of the UCI repository.  
%        % http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
%        load imports-85;
%        Description
%
%        % Extract Price as the response variable and extract non-categorical
%        % variables related to auto construction and performance
%        %
%        X = X(~any(isnan(X(:,1:16)),2),:);
%        Y = X(:,16);
%        Y = log(Y);
%        X = X(:,3:15);
%        predictorNames = {'wheel-base' 'length' 'width' 'height' ...
%            'curb-weight' 'engine-size' 'bore' 'stroke' 'compression-ratio' ...
%            'horsepower' 'peak-rpm' 'city-mpg' 'highway-mpg'};
%
%        % Compute the default sequence of lasso fits.
%        [B,S] = lasso(X,Y,'CV',10,'PredictorNames',predictorNames);
%
%        % Display a trace plot of the lasso fits.
%        axTrace = lassoPlot(B,S);
%        % Display the sequence of cross-validated predictive MSEs.
%        axCV = lassoPlot(B,S,'PlotType','CV');
%        % Look at the kind of fit information returned by lasso.
%        S
%
%        % What variables are in the model corresponding to minimum 
%        % cross-validated MSE, and in the sparsest model within one 
%        % standard error of that minimum.
%        minMSEModel = S.PredictorNames(B(:,S.IndexMinMSE)~=0)
%        sparseModel = S.PredictorNames(B(:,S.Index1SE)~=0)
%
%        % Fit the sparse model and examine residuals.
%        Xplus = [ones(size(X,1),1) X];
%        fitSparse = Xplus * [S.Intercept(S.Index1SE); B(:,S.Index1SE)];
%        corr(fitSparse,Y-fitSparse)
%        figure
%        plot(fitSparse,Y-fitSparse,'o')
%
%        % Consider a slightly richer model. A model with 6 variables may be a 
%        % reasonable alternative.  Find the index for a corresponding fit.
%        df6index = min(find(S.DF==6));
%        fitDF6 = Xplus * [S.Intercept(df6index); B(:,df6index)];
%        corr(fitDF6,Y-fitDF6)
%        plot(fitDF6,Y-fitDF6,'o')         
%         
%        % (2) Run lasso on some random data with 250 predictors
%        %
%        n = 1000; p = 250;
%        X = randn(n,p);
%        beta = randn(p,1); beta0 = randn;
%        Y = beta0 + X*beta + randn(n,1);
%        lambda = 0:.01:.5;
%        [B,S] = lasso(X,Y,'Lambda',lambda);
%        lassoPlot(B,S);
%
%        % compare against OLS
%        %
%        figure
%        bls = [ones(size(X,1),1) X] \ Y;
%        plot(bls,[S.Intercept; B],'.');
%
%        % Run the same lasso fit but restricting the number of
%        % non-zero coefficients in the fitted model.
%        %
%        [B2,S2] = lasso(X,Y,'Lambda',lambda,'DFmax',12);
%
%   See also lassoPlot, ridge, parallelstats.

%   References: 
%   [1] Tibshirani, R. (1996) Regression shrinkage and selection
%       via the lasso. Journal of the Royal Statistical Society,
%       Series B, Vol 58, No. 1, pp. 267-288.
%   [2] Zou, H. and T. Hastie. (2005) Regularization and variable
%       selection via the elastic net. Journal of the Royal Statistical
%       Society, Series B, Vol. 67, No. 2, pp. 301-320.
%   [3] Friedman, J., R. Tibshirani, and T. Hastie. (2010) Regularization
%       paths for generalized linear models via coordinate descent.
%       Journal of Statistical Software, Vol 33, No. 1,
%       http://www.jstatsoft.org/v33/i01.
%   [4] Hastie, T., R. Tibshirani, and J. Friedman. (2008) The Elements
%       of Statistical Learning, 2nd edition, Springer, New York.

%   Copyright 2011-2014 The MathWorks, Inc.

% -------------------------------------- 
%         检查输入参数是否正确
% --------------------------------------

% X是否是一个2维矩阵
if ~ismatrix(X) || length(size(X)) ~= 2 || ~isreal(X)
    error(message('stats:lasso:XnotaReal2DMatrix'));
end

if size(X,1) < 2
    error(message('stats:lasso:TooFewObservations'));
end

% Y是一个向量，并且长度需要和X一样
if ~isvector(Y) || ~isreal(Y) || size(X,1) ~= length(Y)
    error(message('stats:lasso:YnotaConformingVector'));
end

% 如果Y是一个行向量，将其转换为列向量
if size(Y,1) == 1
    Y = Y';
end

% screen（okrows）选择我们可以使用的所有预测和响应
% 但是对于零观察权重可能会进一步减少。
% 进一步检查X，Y直到权重被预先筛选。
okrows = all(isfinite(X),2) & all(isfinite(Y),2);

% --------------------------------------------------------------------
%                        分析和处理可选参数
% --------------------------------------------------------------------

LRdefault = 0.0001;

pnames = { 'weights' 'alpha' 'numlambda' 'lambdaratio' 'lambda' ...
    'dfmax' 'standardize' 'reltol' 'cv' 'mcreps' ...
    'predictornames' 'options' };
dflts  = { []        1       100       LRdefault     []      ...
     []      true          1e-4    'resubstitution'  1 ...
     {}               []};
[weights, alpha, nLambda, lambdaRatio, lambda, ...
    dfmax, standardize, reltol, cvp, mcreps, predictorNames, ParOptions] ...
     = internal.stats.parseArgs(pnames, dflts, varargin{:});

% === 'Alpha'参数 ===

% 必须属于0 < alpha <= 1.
% "0" 对应ridge, "1"对应lasso.
if ~isscalar(alpha) || ~isreal(alpha) || ~isfinite(alpha) || ...
        alpha <= 0 || alpha > 1
    error(message('stats:lasso:InvalidAlpha'))
end

% === 'Weights'参数 ===

if ~isempty(weights)
    % This screen works on weights prior to stripping NaNs and Infs.
    if ~isvector(weights) || ~isreal(weights) || size(X,1) ~= length(weights) || ...
            ~all(isfinite(weights)) || any(weights<0) || sum(weights>0) < 2
        error(message('stats:lasso:InvalidObservationWeights'));
    end
    
    okrows = okrows & weights(:)>0;
    weights = weights(okrows);
    
    % Normalize weights up front.
    weights = weights / sum(weights);
    
    % Below, the convention is that weights is a row vector.
    weights = weights(:)';
end

%用NaN和Infs去除预测值或响应或零观测权重的观测值。
% 
X = X(okrows,:);
Y = Y(okrows);

%去掉之后的X是否是具有多个观察值
if size(X,1) < 2
    error(message('stats:lasso:TooFewObservationsAfterNaNs'));
end

% 如果X有任何常数列，我们希望将它们从坐标下降计算中排除。 相应的系数将返回为零。
% 
% 
constantPredictors = (range(X)==0);
ever_active = ~constantPredictors;

[~,P] = size(X);

% === 'Standardize' option ===

% 获取一个逻辑值.
if ~isscalar(standardize) || (~islogical(standardize) && standardize~=0 && standardize~=1)
    error(message('stats:lasso:InvalidStandardize'))
end

% === 构造lamda序列（在没有提供lamda时） ===

% lambdaMax是系数保证全部为零的惩罚项（λ）。
% 如果命令行不提供lambda序列，我们使用lambdaMax构造默认的lambda序列。
% 我们总是用lambda> lambdaMax跳过计算，
% 因为我们知道lambda> lambdaMax时计算的系数将为零的先验知识。
% 
%
% nullMSE是仅使用一个常数项进行拟合的mse。 当它逐渐清楚我们是过度配合时，它用来终止（越来越少的惩罚）拟合。
% 
% 

[lambdaMax, nullMSE] = computeLambdaMax(X, Y, weights, alpha, standardize);

% 当检测到过度拟合时，与nullMSE（下面计算）一起使用以终止（更少罚分）拟合。
% 
userSuppliedLambda = true;

if isempty(lambda)
    
    % 当检测到过度拟合时，与nullMSE（下面计算）一起使用以终止（更少罚分）拟合。
    % 
    userSuppliedLambda = false;
    
    % 检查Numlambda，应该时一个正整数
    if ~isreal(nLambda) || ~isfinite(nLambda) || nLambda < 1
        error(message('stats:lasso:InvalidNumLambda'));
    else
        nLambda = floor(nLambda);
    end
    
    % 检查LambdaRatio, 应属于 [0,1)之间.
    if ~isreal(lambdaRatio) || lambdaRatio <0 || lambdaRatio >= 1
        error(message('stats:lasso:InvalidLambdaRatio'));
    end
    
    if nLambda==1
        lambda = lambdaMax;
    else
        % 填入数值较小的数字“nLambda”
        if lambdaRatio==0
                lambdaRatio = LRdefault;
                addZeroLambda = true;
        else
            addZeroLambda = false;
        end
        lambdaMin = lambdaMax * lambdaRatio;%%计算lambda的具体值（默认100个），具体物理意义：坐标下降的步进设置
        loghi = log(lambdaMax);             %%开始步子比较大，后面走的小
        loglo = log(lambdaMin);
        logrange = loghi - loglo;
        interval = -logrange/(nLambda-1);
        lambda = exp(loghi:interval:loglo)';
        if addZeroLambda
            lambda(end) = 0;
        else
            lambda(end) = lambdaMin;
        end
    end
    
else

    % Sanity check on user-supplied lambda sequence.  Should be non-neg real.
    if ~isreal(lambda) || any(lambda < 0)
        error(message('stats:lasso:NegativeLambda'));
    end

    lambda = sort(lambda(:),1,'descend');
    
end

% === 'RelTol' parameter ===
%
if ~isscalar(reltol) || ~isreal(reltol) || ~isfinite(reltol) || reltol <= 0 || reltol >= 1
    error(message('stats:lasso:InvalidRelTol'));
end

% === 'DFmax' parameter ===
%
% DFmax is #non-zero coefficients 
% DFmax should map to an integer in [1,P] but we truncate if .gt. P
%
if isempty(dfmax)
    dfmax = P;
else
    if ~isscalar(dfmax)
        error(message('stats:lasso:DFmaxBadType'));
    end
    try
        dfmax = uint32(dfmax);
    catch ME
        mm = message('stats:lasso:DFmaxBadType');
        throwAsCaller(MException(mm.Identifier,'%s',getString(mm)));
    end
    if dfmax < 1
        error(message('stats:lasso:DFmaxNotAnIndex'));
    else
        dfmax = min(dfmax,P);
    end
end

% === 'Mcreps' parameter ===
%
if ~isscalar(mcreps) || ~isreal(mcreps) || ~isfinite(mcreps) || mcreps < 1
    error(message('stats:lasso:MCRepsBadType'));
end
mcreps = fix(mcreps);

% === 'CV' parameter ===
%
if isnumeric(cvp) && isscalar(cvp) && (cvp==round(cvp)) && (0<cvp)
    % cvp is a kfold value. Create a cvpartition to pass to crossval. 
    if (cvp > size(X,1))
        error(message('stats:lasso:InvalidCVforX'));
    end
    cvp = cvpartition(size(X,1),'Kfold',cvp);
elseif isa(cvp,'cvpartition')
    if strcmpi(cvp.Type,'resubstitution')
        cvp = 'resubstitution';
    elseif strcmpi(cvp.Type,'leaveout')
        error(message('stats:lasso:InvalidCVtype'));
    elseif strcmpi(cvp.Type,'holdout') && mcreps<=1
        error(message('stats:lasso:InvalidMCReps'));
    end
elseif strncmpi(cvp,'resubstitution',length(cvp))
    % This may have been set as the default, or may have been
    % provided at the command line.  In case it's the latter, we
    % expand abbreviations.
    cvp = 'resubstitution';
else
    error(message('stats:lasso:InvalidCVtype'));
end
if strcmp(cvp,'resubstitution') && mcreps ~= 1
    error(message('stats:lasso:InvalidMCReps'));
end

if isa(cvp,'cvpartition')
    if (cvp.N ~= size(X,1)) || (min(cvp.TrainSize) < 2)
        % We need partitions that match the total number of observations
        % (after stripping NaNs and Infs and zero observation weights), and
        % we need training sets with at least 2 usable observations.
        error(message('stats:lasso:TooFewObservationsForCrossval'));
    end
end

% === 'PredictorNames' parameter ===
%
% If PredictorNames is not supplied or is supplied as empty, we just 
% leave it that way. Otherwise, confirm that it is a cell array of strings.
%
if ~isempty(predictorNames) 
    if ~iscellstr(predictorNames) || length(predictorNames(:)) ~= size(X,2)
        error(message('stats:lasso:InvalidPredictorNames'));
    else
        predictorNames = predictorNames(:)';
    end
end

% === 'Options' parameter ===
% The 'Options' parameter is passed to crossval for handling.
% crossval will do sanity checking.

% --------------------
%   Lasso模型的回归
% --------------------

% 该结构将包含第二个返回参数。 为永远存在的字段放置持有者，以确保我们想要的结构中的顺序。
% 
%
stats = struct();
stats.Intercept      = [];
stats.Lambda         = [];
stats.Alpha          = alpha;
stats.DF             = [];
stats.MSE            = [];
stats.PredictorNames = predictorNames;

[B,Intercept,lambda,mse] = ...
    lassoFit(X,Y, ...
    weights,lambda,alpha,dfmax,standardize,reltol,lambdaMax,ever_active,userSuppliedLambda,nullMSE);

% 存储每个lambda的非零系数的数量.
df = sum(B~=0,1);

% ---------------------------------------------------------
% 如果需求，使用交叉验证来计算每个lambda的预测均方误差。 
% 
% ---------------------------------------------------------

if ~isequal(cvp,'resubstitution')   
    % Replace dfmax with P, the number of predictors supplied at the
    % command line. dfmax might cause one fold to return empty values, 
    % because no lambda satisfies the dfmax criteria, while other folds 
    % return a numeric value. The lambda sequence has already been 
    % pruned by dfmax, if appropriate, in the call to lassoFit above.
    cvfun = @(Xtrain,Ytrain,Xtest,Ytest) lassoFitAndPredict( ...
        Xtrain,Ytrain,Xtest,Ytest, ...
        lambda,alpha,P,standardize,reltol,ever_active,true);
    if isempty(weights)
        weights = nan(size(X,1),1);
    end
    cvMSE = crossval(cvfun,[weights(:) X],Y, ...
        'Partition',cvp,'Mcreps',mcreps,'Options',ParOptions);
    mse = mean(cvMSE);
    se  = std(cvMSE) / sqrt(size(cvMSE,1));
    minMSE = min(mse);
    minIx = find(mse==minMSE,1);
    lambdaMin = lambda(minIx);
    minplus1 = mse(minIx) + se(minIx);
    seIx = find((mse(1:minIx) <= minplus1),1,'first');
    if isempty(seIx)
        lambdaSE = [];
    else
        lambdaSE = lambda(seIx);
    end
    
    % Deposit cross-validation results in struct for return value.
    stats.SE           = se;
    stats.LambdaMinMSE = lambdaMin;
    stats.Lambda1SE    = lambdaSE;
    stats.IndexMinMSE  = minIx;
    stats.Index1SE     = seIx;
end

% ------------------------------------------
% 按升序lambda排序结果
% ------------------------------------------

nLambda = length(lambda);
reverseIndices = nLambda:-1:1;
lambda = lambda(reverseIndices);
lambda = reshape(lambda,1,nLambda);
B = B(:,reverseIndices);
Intercept = Intercept(reverseIndices);
df = df(reverseIndices);
mse = mse(reverseIndices);
if ~isequal(cvp,'resubstitution')
    stats.SE          = stats.SE(reverseIndices);
    stats.IndexMinMSE = nLambda - stats.IndexMinMSE + 1;
    stats.Index1SE    = nLambda - stats.Index1SE + 1;
end

stats.Intercept = Intercept;
stats.Lambda = lambda;
stats.DF = df;
stats.MSE = mse;
   
end % lasso

% ------------------------------------------
% SUBFUNCTIONS 
% ------------------------------------------

% ===================================================
%                 lassoFitAndPredict() 
% ===================================================

function mse = lassoFitAndPredict(Xtrain,Ytrain,Xtest,Ytest, ...
    lambda,alpha,dfmax,standardize,reltol,ever_active,userSuppliedLambda)
trainWeights = Xtrain(:,1);
if any(isnan(trainWeights))
    trainWeights = [];
end
Xtrain = Xtrain(:,2:end);

[lambdaMax, nullMSE] = computeLambdaMax(Xtrain, Ytrain, trainWeights, alpha, standardize);

[B,Intercept] = lassoFit(Xtrain,Ytrain, ...
    trainWeights,lambda,alpha,dfmax,standardize,reltol,lambdaMax,ever_active,userSuppliedLambda,nullMSE);
Bplus = [Intercept; B];
testWeights = Xtest(:,1);
if any(isnan(testWeights))
    testWeights = ones(size(Xtest,1),1);
end
Xtest = Xtest(:,2:end);
yfit = [ones(size(Xtest,1),1) Xtest] * Bplus;
mse = testWeights'*(bsxfun(@minus,Ytest,yfit).^2) / sum(testWeights);
end

% ===================================================
%       lassoFit() 函数用于求解各lambda值的权值
% ===================================================
function [B,Intercept,lambda,mspe] = ...
    lassoFit(X,Y,weights,lambda,alpha,dfmax,standardize,reltol,lambdaMax,ever_active,userSuppliedLambda,nullMSE)
%
% ------------------------------------------------------
%         对于每个lambda和给定的alpha进行拟合
% ------------------------------------------------------

[N,P] = size(X);
nLambda = length(lambda);

% 如果X有任何常数列，我们希望将它们从坐标下降计算中排除。 相应的系数将返回为零。
% 
% 
constantPredictors = (range(X)==0);
ever_active = ever_active & ~constantPredictors;

% === standardization and weights ===
%
observationWeights = ~isempty(weights);
if ~isempty(weights)
    observationWeights = true;
    weights = weights(:)';
    % Normalize weights up front.
    weights = weights / sum(weights);
end

if ~observationWeights
    muY = mean(Y);
else
    muY = weights*Y;
end
Y0 = bsxfun(@minus,Y,muY);%%将Y向均值靠拢

if standardize
    if ~observationWeights
        % Center and scale
        [X0,muX,sigmaX] = zscore(X,1);%%将X标准化，X0为标准化后的X，另外的分别为均值和标准差
        % 避免初以带0常数预测因子
        sigmaX(constantPredictors) = 1;
    else
        % Weighted center and scale
        muX = weights*X;
        X0 = bsxfun(@minus,X,muX);
        sigmaX = sqrt( weights*(X0.^2) );
        % Avoid divide by zero with constant predictors
        sigmaX(constantPredictors) = 1;
        X0 = bsxfun(@rdivide, X0, sigmaX);
    end
else
    if ~observationWeights
        % Center
        muX = mean(X,1);
        X0 = bsxfun(@minus,X,muX);
        sigmaX = 1;
    else
        % Weighted center
        muX = weights*X;
        X0 = bsxfun(@minus,X,muX);
        sigmaX = 1;
    end
end

% 如果使用观察权重，则对预测矩阵进行加权复制，以节省加权部分回归中的时间。
% 
if observationWeights
    wX0 = bsxfun(@times, X0, weights');
    totalweight = 1;
else
    wX0 = X0;
    totalweight = N;
end

% b将是当前系数估计值，迭代更新。
% 因为我们保留b从一个lambda值到下一个值，
% 所以我们得到事实上的热启动。
b = zeros(P,1);%记录当前的相关性估计，迭代更新

% 预先分配返回的系数矩阵B和截距。
B = zeros(P,nLambda);

active = false(1,P);

for i = 1:nLambda
    
    lam = lambda(i);
    if lam >= lambdaMax
        continue;
    end
    threshold = lam * alpha;
    
    % 坐标下降更新中的分母
    if standardize
        if observationWeights
            shrinkFactor = weights*(X0.^2) + lam*(1 - alpha);
        else
            shrinkFactor = ones(1,P) * (1 + lam*(1 - alpha));
        end
    else
        if observationWeights
            shrinkFactor = weights*(X0.^2) + lam*(1 - alpha);
        else
            shrinkFactor = (1/N) * ones(1,N)*(X0.^2) + lam*(1 - alpha);
        end
    end

    % 迭代坐标下降直至收敛
    while true
        
        bold = b;

        [b,active] = cdescentCycle(X0,wX0,Y0, ...
            b,active,totalweight,shrinkFactor,threshold);%寻找权重和激活集
        
        if norm( (b-bold) ./ (1.0 + abs(bold)), Inf ) < reltol%如果前后两次的权重相差很小，则可以认为该lambda的权重系数已经收敛
%              在激活集合上循环收敛。                          %norm(A，inf):返回A中最大一行和，即max(sum(abs(A’)))
%              完全通过预测变量。 
%              检测是否将预测变量添加到活动集中，如果没有，恢复坐标下降迭代，否则完成迭代。
%              
            bold = b;
            potentially_active = thresholdScreen(X0,wX0,Y0,b,ever_active,threshold);%寻找潜在的活动集
            if any(potentially_active)
                new_active = active | potentially_active;
                [b,new_active] = cdescentCycle(X0,wX0,Y0, ...
                    b,new_active,totalweight,shrinkFactor,threshold);  %%cdescentCycle() 寻找激活集
            else
                new_active = active;
            end

            if isequal(new_active, active)%检查新的激活集和原来的激活集是否一样
                break
            else
                active = new_active;
            end
            
            if norm( (b-bold) ./ (1.0 + abs(bold)), Inf ) < reltol%%如果前后两次的权重相差很小,跳出循环
                break
            end
        end
        
    end
    
    B(:,i) = b;
    
    % 如果达到或超过最大模型尺寸（'DFmax'），则停止。
    if sum(active) > dfmax
        % 输出截断参数B和lambda
        lambda = lambda(1:(i-1));
        B = B(:,1:(i-1));
        break
    end
    
    % 如果我们已经超出了剩余方差百分比的阈值，那么就停下来。
    % 
    if ~userSuppliedLambda
        % 计算当前适应度的MSE（拟合稀疏模型并检查残差）。
        bsig = b ./ sigmaX';%%将其还原成非标准化时的权重
        fit = [ones(size(X,1),1) X] * [(muY-muX*bsig); bsig];%Y=X*beta+c(常数项)
        residuals = bsxfun(@minus, Y, fit);
        if ~observationWeights
            mspe = mean(residuals.^2);
        else
            % This line relies on the weights having been normalized.
            mspe = weights * (residuals.^2);
        end
        if mspe < 1.0e-3 * nullMSE%%如果拟合的mse小于1.0e-3 * nullMSE，退出lambda控制的循环
            lambda = lambda(1:i);
            B = B(:,1:i);
            break
        end
    end
    
end % of lambda sequence

% ------------------------------------------
% 放松居中和缩放（如果有的话）Unwind the centering and scaling (if any)
% ------------------------------------------

B = bsxfun(@rdivide, B, sigmaX');%%元素对应相除
B(~ever_active,:) = 0;
Intercept = muY-muX*B;

% ------------------------------------------
%           计算平均预测平方误差
% ------------------------------------------

BwithI = [Intercept; B];
fits = [ones(size(X,1),1) X]*BwithI;
residuals = bsxfun(@minus, Y, fits);
if ~observationWeights
    mspe = mean(residuals.^2);
else
    % This line relies on the weights having been normalized.
    mspe = weights * (residuals.^2);
end

end %-lassoFit

% ===================================================
%         cdescentCycle() 寻找激活集
% ===================================================

function [b,active] = cdescentCycle(X0, wX0, Y0, ...
    b, active, totalweight, shrinkFactor, threshold)
%r记录了当前预测与Y之间的差距：即残差
r = Y0 - X0(:,active)*b(active,:);

for j=find(active);
    bjold = b(j);
    
    % 回归第j个预测变量的第j个部分残差
    rj = r + b(j)*X0(:,j);%%排除beta j之外的残差
    bj = (wX0(:,j)'*rj) / totalweight;%%确定除j列以外的其他向量预测的残差
    
    % 软阈值
    b(j) = sign(bj) .* max((abs(bj) - threshold), 0) ./ shrinkFactor(j);%检查是否达到软阈值，如果达到，更新j列的权重
    if b(j) == 0
        active(j) = false;
    end
    r = r - X0(:,j)*(b(j)-bjold);
end

end %-cdescentCycle

% ===================================================
%       thresholdScreen() 寻找潜在的活动集
% ===================================================

function potentially_active = thresholdScreen(X0, wX0, Y0, ...
    b, active, threshold)
r = Y0 - X0(:,active)*b(active);
% We don't need the (b.*wX2)' term that one might expect, because it
% is zero for the inactive predictors.
potentially_active = abs(r' *wX0) > threshold;
end %-thresholdScreen

% ===================================================
%      computeLambdaMaX()：计算lambda的最大值
% ===================================================

function [lambdaMax, nullMSE] = computeLambdaMax(X, Y, weights, alpha, standardize)
%
% lambdaMax是系数保证全部为零的惩罚项。
% 其计算公式为lambdaMax=max(abs(XTY))/(N*alpha)，其物理意义：相当于不对其做拟合时的残差最大值
% 
% nullMse是使用一个常数项进行拟合的mse。
% 它在此函数中作为一种便利提供，因为只要计算lambdaMax，就需要在相同的文中计算lambdaMax。
% It is provided in this function as a convenience, because it needs to be calculated 
% in the same context as lambdaMax whenever lambdaMax is calculated.

if ~isempty(weights)
    observationWeights = true;
    weights = weights(:)';        
    % Normalized weights are used for standardization and calculating lambdaMax.
    normalizedweights = weights / sum(weights);
else
    observationWeights = false;
end

[N,~] = size(X);

% 如果我们被要求将预测变量标准化，那么就这样做，因为lambdaMax的计算需要预测因子并使用它们来执行拟合。
%
%

if standardize
    % 如果X有任何常量列，我们希望在正常化差异时防止被零除。
    % 
    constantPredictors = (range(X)==0);

    if ~observationWeights
        % Center and scale
        [X0,~,~] = zscore(X,1);%（x-u）/delta；即标准化
    else
        % Weighted center and scale
        muX = normalizedweights * X;
        X0 = bsxfun(@minus,X,muX);
        sigmaX = sqrt( normalizedweights * (X0.^2) );
        % Avoid divide by zero with constant predictors
        sigmaX(constantPredictors) = 1;
        X0 = bsxfun(@rdivide, X0, sigmaX);
    end
else
    if ~observationWeights
        % Center
        muX = mean(X,1);
        X0 = bsxfun(@minus,X,muX);
    else
        % Weighted center
        muX = normalizedweights(:)' * X;
        X0 = bsxfun(@minus,X,muX);
    end
end

% 如果使用观察权重，则对预测矩阵进行加权复制
% 

if observationWeights
    wX0 = bsxfun(@times, X0, weights');
end

if ~observationWeights
    muY = mean(Y);
else
    muY = weights*Y;
end
% Y0 = bsxfun(@minus,Y,muY);
Y0 = Y - muY;

% 计算允许非零系数的最大lambda
%
if ~observationWeights
    dotp = abs(X0' * Y0);
    lambdaMax = max(dotp) / (N*alpha);
else
    dotp = abs(sum(bsxfun(@times, wX0, Y0)));
    lambdaMax = max(dotp) / alpha;
end

if ~observationWeights
    nullMSE = mean(Y0.^2);
else
    % This works because weights are normalized and Y0 is already
    % weight-centered.
    nullMSE = weights * (Y0.^2);
end
end

function [Xnew, Ynew, consistency] = CollectiveLearning(X,Y,para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   X & Y -- Incomplete data, each row is a sample
%   para -- parameters
%      para.k is the number of nearest neighbors for outlier detection.
%      para.c is the number of instances shared by both views.
%      para.a is the number of instances only in view Y.
%      para.b is the number of instance only in view X.
%      para.type: the type of initialization by filling missing data 
%        'mean':     with average features.
%        'linear':   with linear combination of neighbors.
%        'simWeigh': with the similarity-weighted neighbors.
% Output:
%  consistency -- a score vector whose length is n. 
%      (the smaller, the more likely a sample is to be an outlier)
%  Xnew and Ynew -- new X and Y after filling missing data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note that: 
%     X(1:c,:)       is given
% X = X(c+1:c+a,:)   is missing, we need to solve
%     X(c+a+1:end,:) is given (there are b rows).
%     Y(1:c,:)       is given
% Y = Y(c+1:c+a,:)   is given
%     Y(c+a+1:end,:) is missing, we need to solve (there are b rows).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% settings
addpath(genpath('.\files'));
n = size(Y,1); % n is the number of samples
consistency = ones(n,1);
pre_consistency = ones(n,1);
tol = 1e-7;
maxIter = 1e2;
iter = 0;

%% iteration
while iter < maxIter    
    iter = iter + 1;
    fprintf('\n IterNo. is %d\n', iter);
    
    [Xnew, Ynew, ~, ~] = CL(X, Y, ...
        para.a, para.b, para.c, para.type, pre_consistency);
    [Xsim, ~] = getS(Xnew',para.k,1);
    [Ysim, ~] = getS(Ynew',para.k,1);
    consistency = HSICconsistency3(Xsim, Ysim);
    
    stopC = max(abs(mapminmax(consistency(:)',1e-1,1) ...
        - mapminmax(pre_consistency(:)',1e-1,1)));
    if stopC < tol
       break 
    end    
    pre_consistency = consistency;        
end %end while

end %end this function
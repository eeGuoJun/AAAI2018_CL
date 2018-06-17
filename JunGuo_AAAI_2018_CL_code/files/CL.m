function [Xnew, Ynew, K_x_new, K_y_new] = CL(X, Y, a, b, c, type, Score)
%% max_X,Y  tr[(HXX'HYY').*C] s.t. some rows of X and Y are given
%% Input:
% Incomplete data X and Y, each row represents a sample
%
% c is the number of instances shared by both views.
% a is the number of instances only in view Y.
% b is the number of instance only in view X.
% 
%     X(1:c,:)       is given
% X = X(c+1:c+a,:)   is missing, we need to solve
%     X(c+a+1:end,:) is given (there are b rows)
% 
%     Y(1:c,:)       is given
% Y = Y(c+1:c+a,:)   is given
%     Y(c+a+1:end,:) is missing, we need to solve (there are b rows)
% 
% type: the type of initialization
%    'mean':     fill missing data with average features.
%    'linear':   fill missing data with linear combination of neighbors.
%    'simWeigh': fill missing data with the similarity-weighted neighbors.
%
% Score: a vector whose length is n. Note that C = Score*Score'
%       (the smaller, the more likely it is that the sample is an outlier)
%
%% Output:
% Xnew and Ynew are the new X and Y after filling missing data
% K_x_new = Xnew * Xnew' and K_y_new = Ynew * Ynew'


%% settings
n = size(Y,1);
if size(X,1) ~= n
    error('The number of rows must be the same');
end
H = eye(n) - (1/n)*ones(n);
if (c==n)
    K_x_new = X * X';
    K_y_new = Y * Y';
    Xnew = X;
    Ynew = Y;
    return
end


%% First fill the missing data
switch (type)
    case 'mean'  
        X(c+1:c+a,:) = repmat((sum(X) - sum(X(c+1:c+a,:)))/(c+b), a,1);
        Y(c+a+1:end,:) = repmat(sum(Y(1:c+a,:))/(c+a), b, 1);
    case 'linear'  
        [X, Y] = FillSmp_linearNeighbor(X, Y, a, b, c);
    case 'simWeigh'
        [X, Y] = FillSmp_similarityWeighedNeighbor(X, Y, a, b, c);
    otherwise
        error('The type of initialization must be mean/linear/simWeigh.');
end
X = (normcols(X'))'; % each sample is normalized to unit l2-norm
Y = (normcols(Y'))'; % each sample is normalized to unit l2-norm


%% Second, weighting for different samples
score = Score(:)';
score = mapminmax(score,1e-1,1); % scale
A = diag(score) * Y;
B = diag(score) * X;
A_c = A(1:c,:);
A_a = A(c+1:c+a,:);
% A_b = A(c+a+1:c+a+b,:); % uncertain, we need to solve
B_c = B(1:c,:);
% B_a = B(c+1:c+a,:); % uncertain, we need to solve
B_b = B(c+a+1:c+a+b,:);

K_x = B * B';
K_y = A * A';
L_x = H * K_x * H;
L_y = H * K_y * H;
L_x = 0.5 * (L_x + L_x');
L_y = 0.5 * (L_y + L_y');

K_x_prev = zeros(size(K_x));
K_y_prev = zeros(size(K_y));


%% Third, iteration for completion
count_ = 0;
while ( (max(max(abs(K_x_prev-K_x)))>1e-7 || max(max(abs(K_y_prev-K_y)))>1e-7) && count_<1e2 )
    K_x_prev = K_x;
    K_y_prev = K_y;

	L_x_cb = L_x(1:c, c+a+1:c+a+b);
    L_x_ab = L_x(c+1:c+a, c+a+1:c+a+b);
    L_x_bb = L_x(c+a+1:c+a+b, c+a+1:c+a+b);
	
    A_b = -(L_x_bb)\(L_x_cb')*A_c - (L_x_bb)\(L_x_ab')*A_a;
    A_b = (normcols((diag(1./score(c+a+1:c+a+b))*A_b)'))'; % each sample is normalized to unit l2-norm
    A(c+a+1:c+a+b,:) = diag(score(c+a+1:c+a+b)) * A_b;
	K_y = A * A';
	L_y = H * K_y * H;
	L_y = 0.5 * (L_y + L_y');

    L_y_ca = L_y(1:c, c+1:c+a);
    L_y_aa = L_y(c+1:c+a, c+1:c+a);
    L_y_ab = L_y(c+1:c+a, c+a+1:c+a+b);

    B_a = -(L_y_aa)\(L_y_ca')*B_c - (L_y_aa)\(L_y_ab)*B_b;
    B_a = (normcols((diag(1./score(c+1:c+a))*B_a)'))'; % each sample is normalized to unit l2-norm
    B(c+1:c+a,:) = diag(score(c+1:c+a)) * B_a;
	
    K_x = B * B';
	L_x = H * K_x * H;
	L_x = 0.5 * (L_x + L_x');
    
    if mod(count_, 20) == 1
        fprintf('count_ is %d\n', count_);
    end
    
    count_ = count_ +1;
end

Xnew = diag(1./score) * B;
Ynew = diag(1./score) * A;
K_x_new = 0.5 * (K_x + K_x');
K_y_new = 0.5 * (K_y + K_y');

end

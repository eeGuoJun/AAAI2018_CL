function [Xnew, Ynew] = FillSmp_linearNeighbor(X, Y, a, b, c)
%% Fill the missing data with linear combination of neighbors.

% Incomplete data X and Y, each row represents an instance
% c is the number of instances shared by both views.
% a is the number of instances only in view Y.
% b is the number of instance only in view X.
% X(c+1:c+a,:) are the missing data in X 
% Y(c+a+1:end, :) are the missing data in Y.


if size(X,1) ~= size(Y,1)
    error('The number of rows must be the same');
end

Xnew = X;
Ynew = Y;

if a~=0
    % fill the missing data with linear combination of neighbors.
    Xnew(c+1:c+a,:) = linearNeighbor(Y(c+1:c+a,:)',Y(1:c,:)',X(1:c,:)',5)';
    Ynew(c+a+1:end,:) = linearNeighbor(X(c+a+1:end,:)',X(1:c,:)',Y(1:c,:)',5)';
end

Xnew = (normcols(Xnew'))'; %each sample is normalized to unit l2-norm
Ynew = (normcols(Ynew'))'; %each sample is normalized to unit l2-norm

end


%% =====================================================
function [Smp2] = linearNeighbor(Smp,common,common2,k) 
% For matrices: each column is a sample
Dist = EuDist2(Smp',common',0); % Euclidean distance.^2
[~, idx] = sort(Dist,2);        % sort each row ascend
idx = idx(:,2:k+1)';            % default: not self-connected
Smp2 = zeros(size(common2,1),size(Smp,2));
for ii = 1:size(Smp,2)
   z = common(:,idx(:,ii))-repmat(Smp(:,ii),1,k); % shift ith pt to origin
   C = z'*z;                                      % local covariance
   C = C + eye(k)*eps*trace(C);                   % regularlization
   W = C\ones(k,1);                               % solve Cw=1
   W = W/sum(W);                                  % enforce sum(w)=1
   Smp2(:,ii) = common2(:,idx(:,ii))*W;
end

end
function [outXpaired,outXsingle,outlierInd] = geOutlierData(Xpaired,Xsingle,nOutlier,label)
% generate horizontal outliers
% 
% Input: (For 'Xpaired' and 'Xsingle', each row is a sample.)
%       Xpaired     -View1's samples that have View2 
%       Xsingle     -View1's samples that do not have View2
%		nOutlier    -the number of outliers
%		label       -a nSmp*1 label vector
% Output:
%       outXpaired  -View1's samples that have View2
%       outXsingle  -View1's samples that do not have View2
%       outlierInd  -a nSmp*1 vector with each element: 1-outlier,0-normal


outXpaired = Xpaired;
outXsingle = Xsingle;
outlierInd = zeros(size(label));

[nPaired, ~] = size(Xpaired);
[nSingle, ~] = size(Xsingle);
% PairedLabel = label(1:nPaired);
% SingleLabel = label(nPaired+1:nPaired+nSingle+1);

for i = 1:fix(0.5*nOutlier)
    RAND = randperm(nPaired);
    for j = nPaired:-1:2
        if label(RAND(1))~=label(RAND(j)) && outlierInd(RAND(1))==0 && outlierInd(RAND(j))==0
            outXpaired([RAND(1);RAND(j)],:) = outXpaired([RAND(j);RAND(1)],:);
            outlierInd(RAND(1)) = 1;
            outlierInd(RAND(j)) = 1;
            break
        end            
    end    
end

function consistency = HSICconsistency3(sim_v1, sim_v2)
%% Compute the outlier scores by HSIC
% Input:
%   sim_v1:
%       a n-by-n similarity matrix for the first view of n objects
%   sim_v2:
%       a n-by-n similarity matrix for the second view of n objects
% Output:
%   consistency:
%       An outlier score vector computed by HSIC (The smaller it 
%       is, the more likely it is that the point is an outlier)


%% settings
n = size(sim_v1, 1);
neighborhood_affinity(:, :, 1) = sim_v1;
neighborhood_affinity(:, :, 2) = sim_v2;

%% computing
H = eye(n) - (1/n)*ones(n);
consistency = H * (neighborhood_affinity(:, :, 1) + ...
    neighborhood_affinity(:, :, 1)') * ...
    H * (neighborhood_affinity(:, :, 2) + ...
    neighborhood_affinity(:, :, 2)');
consistency = diag(consistency);
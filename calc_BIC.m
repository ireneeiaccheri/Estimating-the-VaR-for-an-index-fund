function [BIC, pOpt , qOpt] = calc_BIC(data)
%BIC computation

% In the GARCH(p,q) model we rewrite Yt = mu + epst where
% - epst = sigmat * zt , zt iid N(0,1)
% - mu in R

% where sigmat^2 = alpha0 + sum_{1:p} alphai * epst-i^2 + sum_{1:q} betaj *
% sigmat-j^2

% 1. find mu and centered data
tr_data = data(1:1152);
mu = mean(tr_data);
centered_tr_data = tr_data - mu;

% 2. estimate p and q: BIC
% I want to minimize the BIC = -2* logL + m * log(n), where 
%   - logL : log-likelihood of the fitted model
%   - m : number of parameters estimated 
%   - n : number of data points 

n = 1152;

% I create a grid for possible values of p and q
pMax = 5; % Example range for p
qMax = 5; % Example range for q

% % Total number of parameters: alpha0 + p (alphai) + q (betaj)
% m = 1 + length(pMax) * length(q); 

% Initialize BIC
BIC = zeros(pMax, qMax);


% Calculate all the possible BIC values
for p = 1:pMax
    for q = 1:qMax
        try 
            % define the model
            model = garch(p,q);

            % estimate (takes the theoretical model (GARCH), observed data and)
            % find the paramiters that maximize the logLikelihood
            [fit, ~, logL] = estimate(model, centered_tr_data, 'Display', 'off');
            
            % number of parameters
            k = 1 + p + q;

            % Calculate the BIC with the current choice of (p,q)
            BIC(p,q) = -2* logL + k * log(n);

     
        catch
            BIC(p,q) = NaN;
        end
    end
end

% Find the optimal BIC
[minBIC, idx] = min(BIC(:));

% Extract optimal p and q values based on the index
[pOpt, qOpt] = ind2sub(size(BIC), idx);

end
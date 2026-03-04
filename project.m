clear; clc; close all;

% --- Setup------
v = readmatrix('speur3505ydaily');
a = 100;

% initialization of y vector of log-returns and y^2
y = a * diff(log(v));
ysquare = y.^2;

n = length(y);

% plot log-returns
figure(1)
plot(y)
title('Log Returns of S&P Europe 350')
xlabel('Time (Days)')
ylabel('Returns')
grid on

%---- auxiliary function to comment results of test
function helper(test_name, p_value, h_stat, alpha)
    fprintf('\n--- %s ---\n', test_name);
    fprintf('P-Value: %.5f\n', p_value);
    if h_stat == 1
        fprintf('Result: Reject Null Hypothesis (h=1)\n');
        fprintf('Conclusion: Significant evidence AGAINST the assumption.\n');
    else
        fprintf('Result: Fail to Reject Null Hypothesis (h=0)\n');
        fprintf('Conclusion: Insufficient evidence to reject the assumption.\n');
    end
end

%% ---- PROBLEM 1-----------------------------------
%% Autocorrelation, partial autocorrelation and Ljung-Box Test on y_t and y_t^2

% -> Compute autocorrelation and partial autocorrelation
acorr=autocorr(y, 'NumLags', 30);
acorr_squared=autocorr(ysquare, 'NumLags', 30);
partialcorr=parcorr(y, 'NumLags', 30);  

% plot autocorr of Y_t
figure(2)     
autocorr(y, 'NumLags', 30); % Plot the first 30 lags
title('Autocorrelation Function (ACF) - MA Properties');

% plot autocorr of Y_t^2
figure(3)
autocorr(ysquare, 'NumLags', 30);
title('Autocorrelation of Y squared (ACF) - MA Properties');

% plot partial autocorr of Y_t
figure(4)
parcorr(y, 'NumLags', 30);  % Plot the first 30 lags
title('Partial Autocorrelation Function (PACF) - AR Properties');

% ---Test A: Ljung-Box on Raw Returns (Y_t) ---
% H0: No autocorrelation exists in the raw returns
% H1: Autocorrelation exists.
% We often test up to lag 10 or 20.
lags = 15;
[h_lb_raw, p_lb_raw] = lbqtest(y, 'Lags', lags);
fprintf('\n=== Independence Tests (Ljung-Box) ===');
helper(['Ljung-Box on Returns (Lag ' num2str(lags) ')'], p_lb_raw, h_lb_raw, 0.05);

% --- Test B: Ljung-Box on Squared Returns (Y_t^2) ---
% H0: No autocorrelation in variance (No volatility clustering).
% H1: Volatility clustering exists (Data is NOT independent).
[h_lb_sq, p_lb_sq] = lbqtest(ysquare, 'Lags', lags);
helper(['Ljung-Box on Squared Returns (Lag ' num2str(lags) ')'], p_lb_sq, h_lb_sq, 0.05);

%% Normality tests: Lilliefors Test, QQ-plot, histogramm

% --- Test A: Lilliefors Test on  Returns (Y_t)
% H0: The data comes from a normal distribution.
% H1: The data does not come from a normal distribution.
[h_lillie, p_lillie] = lillietest(y);
helper('Lilliefors Test', p_lillie, h_lillie, 0.05);

% --- QQ-Plot
figure(5);
qqplot(y);
title('Q-Q Plot of Log Returns');

% --- Histogramm
mu = mean(y);
sigma = std(y);

figure(6);
histogram(y,30,'Normalization','pdf');
hold on;
x = linspace(min(y), max(y), 100);
norm = normpdf(x, mu, sigma);
plot(x, norm, 'r', 'LineWidth', 2);
title('Histogram with normal curve');
hold off;

%% ---PROBLEM 2------------------------------
%% Setup and Data Loading

% 1. Split Data
n_test  = 100;
n_train = n - n_test;

Y_train = y(1:n_train);      % training data set
Y_test  = y(n_train+1:end);  % test data set

V_test_prev = v(n_train+1:end-1); % V_{t-1} for the test period
V_test_curr = v(n_train+2:end);   % V_t for the test period

% defining the losses' vector
L = V_test_prev - V_test_curr;

% estimating empirical mean and empirical standard deviation
mu_hat = mean(Y_train);
sigma_hat = std(Y_train);

% setting confidence at level 95%
p = 0.05; 

%% VaR and ES computation
% VaRp(X) = V_t-1 * (1 - exp(mu/a + sigma/a * phi^-1(p)))
% ESp(X) = V_t-1 * (1 - 1/p * exp(mu/a + sigma^2/2a^2) * phi(phi^-1)(p) - sigma/a)

% 1. pre-calculate the constant exponent part for efficiency
VaR_cost = 1-exp( (mu_hat + sigma_hat * norminv(p, 0, 1) ) / a );
ES_cost = 1-(1/p)*exp( mu_hat/a + (sigma_hat^2)/(2*a^2)) *normcdf( norminv(p, 0, 1) - (sigma_hat/a) );

% 2. VaR and ES Formulas
VaRp_vec = V_test_prev .* VaR_cost;
ES_vec = V_test_prev .* ES_cost;

%% Breaches detection

% A breach occurs when the actual Loss > VaR
breaches = L > VaRp_vec;        %boolean vector
num_breaches = sum(breaches);
total_days = length(V_test_curr);
expected_breaches = total_days * p;
breach_indices = find(breaches);

% Setting p = 0.05 (or 5%), we expect the actual loss to stay below 
% this line 95% of the time. We expect the loss to breach 
% this line only 5% of the time --> 5 days on 100


fprintf('Expected Breaches (p=%.2f): %.1f\n', p, expected_breaches);
fprintf('Actual Breaches: %d\n', num_breaches); 

if num_breaches > expected_breaches
    fprintf('Conclusion: The model UNDERESTIMATES risk (Too many breaches).\n');
else
    fprintf('Conclusion: The model is conservative or accurate.\n');
end

figure(7)
plot(L, 'k', 'LineWidth', 1); hold on;
plot(VaRp_vec, 'r', 'LineWidth', 1.5);
plot(ES_vec, 'b', 'LineWidth', 1.5);
plot(breach_indices, L(breach_indices), 'r*', 'MarkerSize', 8, 'LineWidth', 1.5);
title('Realized Losses, VaR, ES and Breaches')
legend('Realized Losses', 'VaR (5%)', 'ES (5%)', 'Breaches');
xlabel('Test Days');
ylabel('Value');
grid on;
hold off;

%% Breaches dections with different values of p

p_vec = [0.03 0.04 0.07 0.1]; % confidence levels
n_p = length(p_vec);

figure(8);

for i = 1:n_p
    
    p = p_vec(i);
    
    % Recompute VaR and ES constants
    VaR = 1 - exp( (mu_hat + sigma_hat * norminv(p, 0, 1)) / a );
    ES  = 1 - (1/p) * exp( mu_hat/a + (sigma_hat^2)/(2*a^2) ) ...
              * normcdf( norminv(p, 0, 1) - (sigma_hat/a) );
    
    % Scale by V_{t-1}
    VaRp_vec = V_test_prev .* VaR;
    ES_vec   = V_test_prev .* ES;
    
    % Breaches
    breaches = L > VaRp_vec;
    breach_indices = find(breaches);
    
    % Subplot
    subplot(2,2,i)
    plot(L, 'k', 'LineWidth', 1); hold on;
    plot(VaRp_vec, 'r', 'LineWidth', 1.5);
    plot(ES_vec, 'b', 'LineWidth', 1.5);
    plot(breach_indices, L(breach_indices), 'r*', 'MarkerSize', 6, 'LineWidth', 1.2);
    
    title(sprintf('p = %.2f', p))
    xlabel('Test Days')
    ylabel('Value')
    legend('Realized Loss', 'VaR', 'ES', 'Breaches', 'Location', 'best')
    grid on;
    hold off;
    
end

%% ---- PROBLEM 3-----------------------------------
% BIC: find optimal p and q

[BIC, pOpt , qOpt] = calc_BIC(y);
fprintf('phat = %d\n', pOpt);
fprintf('qhat = %d\n', qOpt);

%% ---- PROBLEM 4-----------------------------------
% Diagnostic check for GARCH model
% --- Setup

% Garch model works in the following way:
% we rewrite the log returns as 
%   Yt = mu + epst
% where 
% - epst is the random noise: epst = sigmat*zt, where zt is a sequence of
%   iid N(0,1)
% - with the choice of p, q = 1, 
%     sigma_t^2 = alpha0 + alpha1 eps_t-1^2 + beta1 sigma_t-1^2


% estimating alpha0, alpha1, beta1 and mu_hat
model = garch(pOpt, qOpt);
model.Offset = NaN;   % we set the mean not to be necessarly equal to zero

[fit, ~, logL] = estimate(model, Y_train, 'Display', 'off');

alpha0 = fit.Constant;
alpha1 = fit.ARCH{1};
beta1 = fit.GARCH{1};
mu_hat = fit.Offset; 

%% Finding the conditional variance sigma_t^2 

condVar = infer(fit, Y_train);
sigma_hat = sqrt(condVar);

standardizedResiduals = (Y_train - mu_hat)./sigma_hat;

% plot the standardize residuals 
figure(9)
plot(1:length(standardizedResiduals), standardizedResiduals, ...
     '.', 'MarkerSize', 10, 'MarkerEdgeColor', 'b')
hold on
yline(0, '--k', 'LineWidth', 1.2)         
yline(2, ':r', 'LineWidth', 1)             
yline(-2, ':r', 'LineWidth', 1)
title('Standardized Residuals')
grid on
xlim([1 length(standardizedResiduals)])

%% Autocorrelation and partial autocorrelation
figure(10);
autocorr(standardizedResiduals, 'NumLags', 30); % Plot the first 30 lags
title('Autocorrelation Function (ACF) of residuals')

figure(11) ;  
autocorr(standardizedResiduals.^2, 'NumLags', 30); % Plot the first 30 lags
title('Autocorrelation Function (ACF) of squared residuals')


%% Lilliefors Test
% --- Lilliefors Test ---
% H0: The data comes from a normal distribution.
% H1: The data does not come from a normal distribution.
[h_lillie, p_lillie] = lillietest(standardizedResiduals);
helper('Lilliefors Test', p_lillie, h_lillie, 0.05);


%% Q-Q Plot and Histogram
figure(13);
qqplot(standardizedResiduals);
title('Q-Q Plot of Standardized Residuals with GARCH');

mu = mean(standardizedResiduals);
sigma = std(standardizedResiduals);

figure(14);
histogram(standardizedResiduals,30,'Normalization','pdf');
hold on;
x = linspace(min(standardizedResiduals), max(standardizedResiduals), 100);
y = normpdf(x, mu, sigma);
plot(x, y, 'r', 'LineWidth', 2);
title('Histogram with normal curve');
hold off;


%% --- PROBLEM 5: Rolling VaR & ES Forecast (Test Set) ---

p = 0.05;                   % Significance Level (5%)
VaR = NaN(n_test, 1);       % Initialize VaR Vector
ES  = NaN(n_test, 1);       % Initialize ES Vector

% 'Y_history' starts as the training set and grows every day
Y_history = Y_train; 

for t = 1:n_test
    
    % re-estimate model using data available up to yesterday
    [estModel, ~, ~] = estimate(model, Y_history, 'Display','off');
    
    % forecast tomorrow's Mean and Volatility (1-step ahead)
    vF = forecast(estModel, 1, 'Y0', Y_history);
    sigmaF = sqrt(vF);        % predicted volatility for tomorrow
    muF    = estModel.Offset; % predicted mean (Offset)
    
    VaRF = 1 - exp( (muF+ sigmaF * norminv(p, 0, 1)) / a );
    ESF  = 1 - (1/p) * exp( muF/a + (sigmaF^2)/(2*a^2) ) ...
              * normcdf( norminv(p, 0, 1) - (sigmaF/a) );
    

    VaR(t) = V_test_prev(t) * VaRF;
    ES(t) = V_test_prev(t) * ESF;
    
    % update the vector by adding today's return
    Y_history = [Y_history; Y_test(t)];
    
end

%% --- BREACHES ---

% count Breaches
breachIdx = L > VaR;       % Logical vector where Loss > VaR
num_breaches = sum(breachIdx);
expected_breaches = p * n_test;

fprintf('\n--- Backtesting Summary ---\n');
fprintf('Total Test Days: %d\n', n_test);
fprintf('VaR Breaches:    %d\n', num_breaches);
fprintf('Expected:        %.1f\n', expected_breaches);

% plot results
figure(13);
plot(L, 'k', 'LineWidth', 1); hold on;
plot(VaR, 'r', 'LineWidth', 1.5);
plot(ES, 'b--', 'LineWidth', 1.5);

% highlight breaches
plot(find(breachIdx), L(breachIdx), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);

title('GARCH Model: Out-of-Sample VaR & ES vs Realized Losses');
xlabel('Test Days');
ylabel('Loss');
legend('Realized Loss', 'VaR 5%', 'ES 5%', 'Breaches', 'Location', 'Best');
grid on;

%% --- PROBLEM 6: Rolling VaR & ES Forecast with GARCH t-Student (Test Set) ---

p = 0.05;                   % Significance Level (5%)
VaR_tStudent = NaN(n_test, 1);       % Initialize VaR Vector
ES_tStudent  = NaN(n_test, 1);       % Initialize ES Vector

% 'Y_history' starts as the training set and grows every day
Y_history = Y_train; 

model_tStudent = garch(pOpt, qOpt);
model_tStudent.Offset = NaN;
model_tStudent.Distribution = 't';

for t = 1:n_test
    
    % re-estimate model using data available up to yesterday
    [estModel_tStudent, ~, ~] = estimate(model_tStudent, Y_history, 'Display','off');
    
    % forecast tomorrow's Mean and Volatility
    vF = forecast(estModel_tStudent, 1, 'Y0', Y_history);
    sigmaF = sqrt(vF);        % predicted volatility for tomorrow
    muF    = estModel_tStudent.Offset; % predicted mean (Offset)
    nuF = estModel_tStudent.Distribution.DoF;

    VaRF = 1 - exp( (muF+ sigmaF * tinv(p, nuF)) / a );
    ESF  = 1 - (1/p) * exp( muF/a + (sigmaF^2)/(2*a^2) ) ...
              * normcdf( norminv(p, 0, 1) - (sigmaF/a) );
    

    VaR_tStudent(t) = V_test_prev(t) * VaRF;
    ES_tStudent(t) = V_test_prev(t) * ESF;
    
    % update the vector by adding today's return
    Y_history = [Y_history; Y_test(t)];
    
end

%% --- BREACHES ---
% comparison between gaussian and t-student GARCH-model

% count Breaches
breachIdx = L > VaR_tStudent;       % Logical vector where Loss > VaR
num_breaches = sum(breachIdx);
expected_breaches = p * n_test;

fprintf('\n--- Backtesting Summary ---\n');
fprintf('Total Test Days: %d\n', n_test);
fprintf('VaR Breaches:    %d\n', num_breaches);
fprintf('Expected:        %.1f\n', expected_breaches);

% plot results
figure(13);
plot(L, 'k', 'LineWidth', 1); hold on;
plot(VaR_tStudent, 'r', 'LineWidth', 1.5);
plot(VaR, 'g', 'LineWidth', 1.5)

% highlight breaches
plot(find(breachIdx), L(breachIdx), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);

title('GARCH Model: VaR comparison between t-Student and Gaussian');
xlabel('Test Days');
ylabel('Loss');
legend('Realized Loss', 'VaR t-Student', 'VaR Gaussian', 'Breaches t-Student', 'Location', 'Best');
grid on;
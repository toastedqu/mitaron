
%% Read data from excel sheet with data and agent forecasts
yI = xlsread('^N225.xlsx'); % Target data (quarterly inflation)
a = xlsread('^N225_mean.xlsx'); % Agent mean
A = xlsread('^N225_var.xlsx'); % Agent variance
n = xlsread('^N225_dof.xlsx'); % Agent degrees of freedom
T = length(yI); % Total time in analysis
J = size(a,2); % Number of agents
p = J+1; % Number of agents + intercept
K = 100; % Number of forecast periods

%% Set priors
delta = [0.95 0.99]; % Discount factor [state observation] variance
m_0 = [0 ones(1,J)*1/J]; % Prior on mean of BPS coefficients
C_0 = eye(p)*1; % Prior on covariance of BPS coefficients
n_0 = 1/(1-delta(2)); % Prior on BPS degrees of freedom
s_0 = 0.01; % Prior on BPS observation variance

%% Burn_in and MCMC
burn_in = 100;
mcmc_iter = 200;

%% Run BPA
E_BPS = zeros(mcmc_iter,K); % Posterior BPS mean
V_BPS = zeros(mcmc_iter,K); % Posterior BPS variance
error = zeros(mcmc_iter,K);
mlike = zeros(mcmc_iter,K);
ak_results = cell(K,1); % Posterior BPS forecast coefficient mean
Rk_results = cell(K,1); % Posterior BPS forecast coefficient variance
vt_results = cell(K,1); % Posterior BPS forecast observation variance
nt_results = cell(K,1); % Posterior BPS forecast degrees of freedom
nu = zeros(K,1);
std_var = @(x) (x+x')/2;
for t=50:T-1
    y = yI(1:t);
    a_j = a(1:t,:);
    A_j = A(1:t,:);
    n_j = n(1:t+1,:);
    [a_k,R_k,v_k,n_k,theta_post,X_post] = ...
        BPS(y,a_j,A_j,n_j,delta,m_0,C_0,n_0,s_0,burn_in,mcmc_iter);
    ak_results{t,1} = a_k;
    Rk_results{t,1} = R_k;
    vt_results{t,1} = v_k;
    nt_results{t,1} = n_k;
    nu(t,1) = n_k;
    for i=1:mcmc_iter
        % sample x(t+1)
        lambda = sqrt((0.5*delta(2)*n(t+1)/randg(delta(2)*n(t+1)/2)));
        x_t = [1 a(t+1,:)+lambda*randn(1,length(a(t+1,:)))*chol(std_var(diag(A(t+1,:))))];
        % compute aggregated mean and variance
        E_BPS(i,t) = x_t*a_k(i,:)';
        V_BPS(i,t) = x_t*R_k(p*i-(p-1):p*i,:)*x_t'+v_k(i,:);
        error(i,t) = yI(t+1)-E_BPS(i,t);
%         mlike(i,t) = exp(log(gamma(0.5*(nu(t,1)+1)))-log(gamma(0.5*nu(t,1)))...
%             -0.5*log(pi*nu(t,1)*V_BPS(i,t))-(0.5*(nu(t,1)+1))*log(1+1/(nu(t,1)*V_BPS(i,t))*(yI(t+1)-E_BPS(i,t))^2));
    end
end
BPS_error = mean(error(:,end-(K-1):end),1); % Average error of BPS
w = (1:length(BPS_error));
BPS_rmse = ((cumsum(BPS_error(:,1:end).^2))./w).^(1/2); % RMSFE of BPS
% BPS_mlike = cumsum(log(mean(mlike(:,end-(K-1):end))')); % Marginal likelihood of BPS
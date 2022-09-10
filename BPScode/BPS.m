function [a_k,R_k,v_k,n_k,theta_post,X_post] = BPS(y,a_j,A_j,n_j,delta,m_0,C_0,n_0,s_0,burn_in,mcmc_iter)
% Bayesian Predictive Synthesis (McAlinn & West, 2019, Journal of Econometrics 210: 155-169):
%
%  Synthesis Function:      
%           y_t = x_t'\theta_t + \nu_t
%      \theta_t = \theta_{t-1} + \omega_t
%         \nu_t \sim Normal(0,v_t)
%      \omega_t \sim Normal(0,v_tW_t)
%
%  Forecasts: 
%       x_{j,t} \sim t(a_{j,t},A_{j,t},n_{j,t})
%
%  inputs:
%    y: target time series data from t=1:T
%    a_j: TxJ matrix of mean of agent forecast t=1:T
%    A_j: TxJ matrix of variance of agent forecast t=1:T
%    n_j: TxJ matrix of d.o.f. of agent forecast t=1:T
%    delta: discount rate for [state obs_var]
%    m0 --  1xJ vector prior mean for state
%    C0 --  JxJ prior var matrix
%    n0 --  prior d.o.f.
%    s0 --  prior of obs var
%    burn_in, mcmc_iter: number of burn-in/MCMC iterations
%  outputs:
%    a_k: forecast mean of \theta_{T+1}
%    R_k: forecast variance of \theta_{T+1}
%    v_k: forecast obs var
%    n_k: forecast d.o.f.
%    theta_post: posterior \theta_{1:T}
%    X_post: posterior sample of agent forecast
%
%  ? 2017, Kenichiro McAlinn, All rights reserved.

%% initial settings
std_var = @(x) (x+x')/2;

mcmc_iter = burn_in+mcmc_iter;
T = length(y);
p_x = size(a_j,2);
p = p_x+1;

m_t = zeros(T+1,p);
C_t = zeros((T+1)*p,p);
n_t = zeros(T+1,1);
s_t = zeros(T+1,mcmc_iter);
v_t = zeros(T,mcmc_iter);
a_t = zeros(T,p);
R_t = zeros(T*p,p);
f_t = zeros(T,1);
q_t = zeros(T,1);
phi_t = zeros(T,p_x);
X_t = zeros(T,p_x*(mcmc_iter+1));
theta_t = zeros(T,p*mcmc_iter);
a_k = zeros((mcmc_iter),p);
R_k = zeros(p*(mcmc_iter),p);
v_k = zeros((mcmc_iter),1);
n_k = zeros(1,1);

d = delta(1);
beta = delta(2);

m_t(1,:) = m_0;
C_t(1:p,:) = C_0;
n_t(1) = n_0;
s_t(1,:) = s_0;

for t=1:T
    phi_t(t,:) = ((0.5*beta*n_j(t,:)./randg(beta*n_j(t,:)/2)));
    X_t(t,1:p_x) = a_j(t,:)...
        +randn(1,length(a_j(t,:)))*chol(std_var(diag(phi_t(t,:).*A_j(t,:))));
end

%% MCMC Sampler
for i=1:mcmc_iter
    % forward-filter
    for t=1:T
        F_t = [1 X_t(t,p_x*i-(p_x-1):p_x*i)]';
        % prior for time t
        a_t(t,:) = m_t(t,:);
        R_t(p*t-(p-1):p*t,:) = C_t(p*t-(p-1):p*t,:)/d;
        % predict time t
        f_t(t) = F_t'*a_t(t,:)';
        q_t(t) = F_t'*(C_t(p*t-(p-1):p*t,:)*F_t/d)+s_t(t,i);
        % compute forcast error and adaptive vector
        e_t = y(t)-f_t(t);
        A_t = R_t(p*t-(p-1):p*t,:)*F_t/q_t(t);
        % posterior for time t
        n_t(t+1) = beta*n_t(t)+1;
        r_t = (beta*n_t(t)+e_t^2/q_t(t))/n_t(t+1);
        s_t(t+1,i) = r_t*s_t(t,i);
        m_t(t+1,:) = a_t(t,:)+(A_t*e_t)';
        C_t(p*(t+1)-(p-1):p*(t+1),:) = ...
            std_var(r_t*(R_t(p*t-(p-1):p*t,:)-q_t(t)*(A_t*A_t')));  
    end
    % sample theta at T
    v_t(end,i) = 1/gamrnd(n_t(end)/2,2/(n_t(end)*s_t(end,i)));
    theta_t(T,p*i-(p-1):p*i) = m_t(end,:)...
        +randn(1,length(m_t(end,:)))*chol(std_var(C_t(end-(p-1):end,:)*(v_t(end,i)/s_t(end,i))));
    % theta at T+1
    n_k = beta*n_t(end)+1;
    v_k(i,1) = 1/gamrnd(beta*n_t(end)/2,2/(beta*n_t(end)*s_t(end,i)));
    a_k(i,:) = m_t(end,:);
    R_k(p*i-(p-1):p*i,:) = (C_t(end-(p-1):end,:)/d)*(v_k(i,1)/s_t(end,i));
    % backward-sampler
    for t=T-1:-1:1
        v_t(t,i) = 1/(1/v_t(t+1,i)*beta...
            +gamrnd((1-beta)*n_t(t+1)/2,2/(n_t(t+1)*s_t(t+1,i))));
        m_star_t = m_t(t+1,:)'+d*(theta_t(t+1,p*i-(p-1):p*i)'-a_t(t+1,:)'); 
        C_star_t = C_t(p*(t+1)-(p-1):p*(t+1),:)*(1-d)*(v_t(t,i)/s_t(t+1,i)); 
        theta_t(t,p*i-(p-1):p*i) = m_star_t'...
            +randn(1,length(m_star_t))*chol(std_var(C_star_t));
    end
    % sample X_t
     for t=1:T
         A_st = diag(phi_t(t,:).*A_j(t,:));
         a_st = a_j(t,:);
         theta_p = theta_t(t,p*i-(p-2):p*i);
         theta_1 = theta_t(t,p*i-(p-1));
         sigma = theta_p*A_st/(v_t(t,i)+theta_p*A_st*theta_p');
         a_star = a_st+sigma*(y(t)-(theta_1+theta_p*a_st'));
         A_star = std_var(A_st-A_st*theta_p'*sigma);
         X_t(t,p_x*(i+1)-(p_x-1):p_x*(i+1)) = a_star+randn(1,length(a_star))...
             *chol(std_var(A_star));
         phi_t(t,:) = ((0.5*(n_j(t,:)+1)./randg((n_j(t,:)+...
             (X_t(t,p_x*(i+1)-(p_x-1):p_x*(i+1))-a_st).^2./A_j(t,:))/2)));
     end
end
% save results
theta_post = theta_t(:,burn_in*p_x+1:end);
X_post = X_t(:,burn_in*p_x+1:end);
a_k = a_k(burn_in+1:end,:);
R_k = R_k(p*burn_in+1:end,:);
v_k = v_k(burn_in+1:end,:);
end
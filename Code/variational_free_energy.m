%% Residuals & MSE
FC_emp_all = load('data/results/FC_emp_all.mat'); 
FC_emp_all = FC_emp_all.FC_emp_all; 
FC_matrices_geo = load('data/results/FC_geom.mat'); 
FC_matrices = FC_matrices_geo.FC_matrices; 

res1 = FC_emp_all(:,:,1) - FC_matrices(:,:,1); 
res_vec1 = res1(:); 

res2 = FC_emp_all(:,:,2) - FC_matrices(:,:,2); 
res_vec2 = res2(:); 

res3 = FC_emp_all(:,:,3) - FC_matrices(:,:,3); 
res_vec3 = res3(:); 

mse1 = mean(res_vec1(:).^2); 
mse2 = mean(res_vec2(:).^2); 
mse3 = mean(res_vec3(:).^2); 

num_subjects = size(FC_emp_all, 3); 
%% Likelihood

all_residuals = cat(3, res1, res2, res3);
all_residuals = all_residuals(:); % Flatten the array into a single column

overall_mse = mean(all_residuals(:).^2); 

overall_likelihood = likelihood(overall_mse, all_residuals); 

% Compute the log-likelihood for the aggregated data
overall_log_likelihood = log_likelihood(overall_mse, all_residuals);

%% KL divergence

num_subjects = size(FC_emp_all, 3); 
KL_divergence = zeros(num_subjects, 1); 

for i = 1:num_subjects
    FC_emp = FC_emp_all(:,:,i); 
    FC_recon = FC_matrices(:,:,i); 

    mu_emp = mean(FC_emp(:)); 
    sigma_emp = std(FC_emp(:)); 

    mu_recon = mean(FC_recon(:)); 
    sigma_recon = std(FC_recon(:)); 

    KL_divergence(i) = log(sigma_recon / sigma_emp) + (sigma_emp^2 + (mu_emp - mu_recon)^2 / (2 * sigma_recon^2) - 0.5); 
end

KL_edr = mean(KL_divergence); 

%% Variational free energy
FC_matrices_geo = load('data/results/FC_geom.mat'); 
FC_matrices_geo = FC_matrices_geo.FC_matrices; 

FC_matrices_edr = load('data/results/FC_edr.mat'); 
FC_matrices_edr = FC_matrices_edr.FC_matrices; 

FC_emp_all = load('data/results/FC_emp_all.mat'); 
FC_emp_all = FC_emp_all.FC_emp_all; 

num_subjects = size(FC_emp_all, 3); 

for subj = 1:num_subjects
    emp = FC_emp_all(:,:,subj); 
    mu_emp(subj) = mean(emp(:)); 
    sigma_emp(subj) = std(emp(:)); 
    
    geo = FC_matrices_geo(:,:,subj); 
    mu_geo(subj) = mean(geo(:)); 
    sigma_geo(subj) = std(geo(:)); 

    edr = FC_matrices_edr(:,:,subj); 
    mu_edr(subj) = mean(edr(:)); 
    sigma_edr(subj) = std(edr(:)); 
end

% Initialize arrays to store results for each subject
mu_variational = zeros(1, num_subjects);
sigma2_variational = zeros(1, num_subjects);

num_initializations = 5; % Number of random initializations per subject
scale_factor = 0.0001; % Adjust this as needed for your specific problem

for subj = 1:num_subjects
    FC_geo = FC_matrices_geo(:,:,subj); 
    FC_geo = FC_geo(:);

    mu_prior = mu_emp(subj);
    sigma2_prior = sigma_emp(subj)^2;

      % Define the optimization problem
    problem = createOptimProblem('fmincon', 'objective', ...
        @(params) -elbo(params, FC_geo, mu_prior, sigma2_prior), ...
        'x0', [mu_prior, sigma2_prior], ...
        'lb', [-Inf, 0], ...  % Lower bounds (assuming variance > 0)
        'ub', [Inf, Inf], ... % Upper bounds
        'options', optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'off'));

    % Initialize GlobalSearch
    gs = GlobalSearch;

    % Run GlobalSearch
    [optimized_params, fval] = run(gs, problem);

    % Update parameters after optimization
    mu_variational(subj) = optimized_params(1);
    sigma2_variational(subj) = optimized_params(2);

    % Update parameters after optimization for the best result
    mu_variational(subj) = best_params(1);
    sigma2_variational(subj) = best_params(2);

    % Plot the optimized normal distribution for the best result
    figure(subj);
    x_range = linspace(mu_variational(subj) - 3*sqrt(sigma2_variational(subj)), mu_variational(subj) + 3*sqrt(sigma2_variational(subj)), 100);
    pdf_values = normpdf(x_range, mu_variational(subj), sqrt(sigma2_variational(subj)));
    plot(x_range, pdf_values, 'LineWidth', 2);
    xlabel('X'); ylabel('Probability Density');
    title(sprintf('Optimized Normal Distribution - Subject #%i', subj));
    grid on;

    % Calculate the ELBO for the current subject
    elbo_values(subj) = elbo(optimized_params, FC_geo, mu_prior, sigma2_prior);

    % VFE is the negative of the ELBO
    vfe_values(subj) = -elbo_values(subj);

    model_evidence(subj) = exp(elbo_values(subj));

end

%%
%% Assumptions and Definitions

% Define the prior distribution parameters (mean and variance)
% For example, if you assume a Gaussian prior with mean 0 and variance 1
for subj = 1:num_subjects
    FC_geo = FC_matrices_geo(:,:,subj); 
    FC_geo = FC_geo(:); 

    mu_prior = mu_emp(subj);
    sigma2_prior = sigma_emp(subj);

% Define initial parameters for the variational distribution (mean and variance)
% These are initial guesses and will be optimized
    mu_variational = 0.1; % Example initial value
    sigma2_variational = 0.1; % Example initial value
end

%% Variational Inference Process
% Optimization process to find the best parameters for the variational distribution
% This can be done using MATLAB's optimization functions such as fminunc

% Define the objective function for optimization (ELBO)
    objective_function = @(params) -elbo(params, FC_geo, mu_prior, sigma2_prior);

% Initial parameters for the optimization [mu, sigma^2]
    initial_params = [mu_variational, sigma2_variational];

% Run the optimization
    optimized_params = fminunc(objective_function, initial_params);

% Update the variational distribution parameters based on optimization
    mu_variational = optimized_params(1);
    sigma2_variational = optimized_params(2);

    elbo_values = elbo(optimized_params, FC_geo, mu_prior, sigma2_prior);

%% Plot
% After obtaining mu_variational and sigma2_variational from the optimization
figure(subj); 
% Define the range for x-axis (typically 3 standard deviations around the mean)
x_range = linspace(mu_variational - 3*sqrt(sigma2_variational), mu_variational + 3*sqrt(sigma2_variational), 100);

% Calculate the PDF values
pdf_values = normpdf(x_range, mu_variational, sqrt(sigma2_variational));

% Plot the distribution
figure(1); % Create a new figure
plot(x_range, pdf_values, 'LineWidth', 2);
xlabel('X'); ylabel('Probability Density');
title('Optimized Normal Distribution');
grid on;
%%
mean_model_evidence = mean(model_evidence); 
%%
mean_mu = mean(mu_variational); 
mean_sigma2 = mean(sigma2_variational); 
mean_elbo_edr = mean(elbo_values); 
mean_elbo_geo = mean(elbo_values); 

%%
bayes_factor = mean_elbo_geo/mean_elbo_edr; 

%%

figure();
x_range = linspace(mean_mu - 3*sqrt(mean_sigma2), mean_mu + 3*sqrt(mean_sigma2), 100);
pdf_values = normpdf(x_range, mean_mu, sqrt(mean_sigma2));
plot(x_range, pdf_values, 'LineWidth', 2);
xlabel('X'); ylabel('Probability Density');
title(sprintf('Optimized Normal Distribution - Mean'));
grid on;

%%
x_range_emp = linspace(mu_prior - 3*sqrt(sigma2_prior), mu_prior + 3*sqrt(sigma2_prior), 100);

% Calculate the PDF values
pdf_values_emp = normpdf(x_range, mu_prior, sqrt(sigma2_prior));

% Plot the distribution
figure(2); % Create a new figure
plot(x_range_emp, pdf_values_emp, 'LineWidth', 2);
xlabel('X'); ylabel('Probability Density');
title('Empirical normal distribution');
grid on;
%%
figure();
hist(res_vec1);
%% Calculate the VFE (negative ELBO) using optimized parameters
vfe = -elbo([mu_variational, sigma2_variational], all_residuals, mu_prior, sigma2_prior);

% Display the VFE
disp(['Variational Free Energy: ', num2str(vfe)]);

%% Required Functions

% Function to compute ELBO
function value = elbo(params, data, mu_prior, sigma2_prior)
    mu_var = params(1);
    sigma2_var = params(2);
    
    % KL Divergence between variational and prior distributions
    kl_div = 0.5 * (log(sigma2_prior / sigma2_var) + ...
                    (sigma2_var + (mu_var - mu_prior)^2) / sigma2_prior - ...
                    1);
    
    % Expected log-likelihood under the variational distribution
    % This part depends on your specific model and might require numerical integration
    % For simplicity, let's assume it's a Gaussian likelihood
    expected_log_likelihood = -0.5 * log(2 * pi * sigma2_var) - ...
                              0.5 * sum((data - mu_var).^2) / sigma2_var;
    
    % ELBO is the expected log-likelihood minus the KL divergence
    value = expected_log_likelihood - kl_div;
end

% Define a function to compute the likelihood
function L = likelihood(mse, residuals)
    n = numel(residuals); % Number of data points
    sigma_squared = mse; % Variance estimate from MSE
    L = (1 / sqrt(2 * pi * sigma_squared))^n * exp(-sum(residuals.^2) / (2 * sigma_squared));
end

% Define a function to compute the log-likelihood
function logL = log_likelihood(mse, residuals)
    n = numel(residuals); % Number of data points
    sigma_squared = mse; % Variance estimate from MSE
    logL = -0.5 * n * log(2 * pi * sigma_squared) - sum(residuals.^2) / (2 * sigma_squared);
end
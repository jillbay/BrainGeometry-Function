% =========================================================================
%           Hybrid model combining geometric & EDR properties
% =========================================================================

%% Load and setup data
% Load in data
Lnorm = load('data/Lnorm_matrix.mat'); 
LBO = load('laplacian_matrix.mat'); 
Lnorm = Lnorm.Lnorm; 
LBO = LBO.laplacian;

addpath(genpath('functions_matlab'));

surface_interest = 'fsLR_32k';
hemisphere = 'lh';
mesh_interest = 'midthickness';

[vertices, faces] = read_vtk(sprintf('data/template_surfaces_volumes/%s_%s-%s.vtk', surface_interest, mesh_interest, hemisphere));
surface_midthickness.vertices = vertices';
surface_midthickness.faces = faces';

% Load cortex mask
cortex = dlmread(sprintf('data/template_surfaces_volumes/%s_cortex-%s_mask.txt', surface_interest, hemisphere));
cortex_ind = find(cortex);

% Normalize laplacians: 
% LBO = LBO/norm(LBO); 
% Lnorm = Lnorm/norm(Lnorm); 

% Since I want to prseserve the intrinsic properties of both matrices,
% avoiding normalization is a good choice, since normalization could alter
% the characteristics of each graph's Laplcain. 

hemisphere = 'lh';
num_modes = 200;

num_subjects = 255; 

data = load(sprintf('data/empirical/S255_tfMRI_ALLTASKS_raw_%s', hemisphere));
data_to_reconstruct = data.zstat;
data_recon= data_to_reconstruct.motor_cue_avg; 
%data_recon = data_to_reconstruct(:,1); % select only 1 subject now
%%
num_vertices = size(data_recon, 1); 

w1 = 10; 
w2 = 1; 

num_subjects = size(data_recon, 2); 
    total_MSE = zeros(num_subjects);

    % compute hybrid Laplacian: 
    Lcombined = w1 * LBO + w2 * Lnorm; 

    % compute the eigenmodes:
    [eig_vec_temp, ~] = eig(Lcombined); 
    eig_vec_temp = eig_vec_temp(:, 1:num_modes); 
    eig_vec = zeros(num_vertices, num_modes); 
    eig_vec(cortex_ind,:) = eig_vec_temp(cortex_ind,1:num_modes); 

    % reconstruct dataand calculate objective
    recon_beta_task = zeros(num_modes, num_modes, num_subjects); 
    for subj = 1:num_subjects
        for mode = 1:num_modes
            basis = eig_vec(cortex_ind, 1:mode); 
            recon_beta_task(1:mode, mode, subj) = calc_eigendecomposition(data_recon(cortex_ind, subj), basis, 'matrix');
        end
    end
    
    data_plot = zeros(num_vertices, num_subjects);
    minimize = zeros(1, num_subjects);
    MSE = zeros(num_vertices, num_subjects);
    N = num_modes; 
    recon_beta_squeeze = squeeze(recon_beta_task); 

    for subj = 1:num_subjects
        disp(['Size eig_vec: ', mat2str(size(eig_vec))]);
        disp(['Size recon_beta_task: ', mat2str(size(recon_beta_task))]);
        disp(['Size data_plot: ', mat2str(size(data_plot))]);

        data_plot(:,subj) = eig_vec(:,1:N) * (recon_beta_squeeze(1:N,N, subj));

        minimize(:,subj) = nanmean(data_recon(:,subj) - data_plot(:,subj));

        for i = 1:num_vertices
            MSE(i, subj) = nanmean((data_recon(i,subj) - data_plot(i,subj))).^2; 
        end
        total_MSE(subj) = nansum(MSE(:, subj));
    end
    
    total = sum(total_MSE); 
    % Ensure obj is a scalar
    disp(['Total MSE: ', num2str(total)]);


%%
num_subjects = 255;
for subj = 1:num_subjects
    data_to_plot(:,subj) = eig_vec(:,1:N) *recon_beta_task(1:N,N,subj); 
end
%%
N = 200; 
surface_to_plot = surface_midthickness; 
medial_wall = find(cortex==0); 
with_medial = 1; 

    fig = draw_surface_bluewhitered_dull(surface_to_plot, data_to_plot(:,5), hemisphere, medial_wall, with_medial); 
    fig.Name = sprintf('tfMRI reconstruction for task - surface map using %i modes', N);

    %%
data_to_reconstruct_avg = nanmean(data_to_reconstruct,2); 

    % reconstruct the average data using the first N modes
    data_to_plot_avg = eig_vec(:,1:N) * calc_eigendecomposition(data_to_reconstruct_avg(cortex_ind), eig_vec(cortex_ind, 1:N), 'matrix'); 

    fig = draw_surface_bluewhitered_dull(surface_to_plot, data_to_plot_avg, hemisphere, medial_wall, with_medial); 
    fig.Name = sprintf('tfMRI reconstruction for task - surface map using %i modes', N);


%% Gradient descent 
% Initialize weights and learning rate 
w1 = 10; 
w2 = 1;
alpha = 0.01; % learning rate
iter = 100; % number of iterations
prevObjective = Inf; 
tolerance = 1e-2; 

% w1 = 10 and w2 = 1 yields avg_MSE = 1.4755
% only geometric method (w1 = 1, w2 = 0) yields avg_MSE = 1.4888


for i = 1:iter
    [dw1, dw2] = computeGradient([w1, w2], LBO, Lnorm, data_to_reconstruct, num_modes, num_vertices, cortex_ind); 

    % update weights:
    w1 = w1 - alpha * dw1; 
    w2 = w2 - alpha * dw2; 

    % calculate the current objective function value:
    currentObjective = computeObjective([w1, w2], LBO, Lnorm, data_to_reconstruct, num_modes, num_vertices, cortex_ind); 

    % Print the objective function value every few iterations
    if mod(i, 100) == 0
        disp(['Iteration ', num2str(i), ': Objective = ', num2str(currentObjective)]); 
    end

    % check for convergence: 
    if abs(prevObjective - currentObjective) < tolerance
        disp(['Converged at iteration ', num2str(i)]); 
        break;
    end

    prevObjective = currentObjective; 
end

% final weights: 
disp(['Final weights: w1 = ', num2str(w1), ', w2 = ', num2str(w2)]); 

% starting w1 = w2 = 0.5 -> converged at iteration 2 with weights w1 = w2 = 0.5
% starting w1 = 0.9, w2 = 0.1 -> converged at iteration 2 with weights w1 =
% 0.9 and w2 = 0.1

%% Function to compute gradient
function [dw1, dw2] = computeGradient(w, LBO, Lnorm, data_recon, num_modes, num_vertices, cortex_ind)
    delta = 1e-2; % small change for numerical differentiation

    % original:
    original = computeObjective(w, LBO, Lnorm, data_recon, num_modes, num_vertices, cortex_ind); 

    % w1 slightly increased
    w1_incr = [w(1) + delta, w(2)]; 
    obj_w1 = computeObjective(w1_incr, LBO, Lnorm, data_recon, num_modes, num_vertices, cortex_ind); 

    % w2 slightly increased
    w2_incr = [w(1), w(2) + delta]; 
    obj_w2 = computeObjective(w2_incr, LBO, Lnorm, data_recon, num_modes, num_vertices, cortex_ind); 

    % numerical derivatives
    dw1 = (obj_w1 - original) / delta; 
    dw2 = (obj_w2 - original) / delta; 

    disp(['dw1: ', num2str(dw1)]);
    disp(['dw2: ', num2str(dw2)]);
end

%% Function to compute the objective
function total_MSE = computeObjective(w, L_geo, L_edr, data_recon, num_modes, num_vertices, cortex_ind)

    num_subjects = size(data_recon, 2); 
    total_MSE = zeros(num_subjects);

    % compute hybrid Laplacian: 
    Lcombined = w(1) * L_geo + w(2) * L_edr; 

    % compute the eigenmodes:
    [eig_vec_temp, ~] = eig(Lcombined); 
    eig_vec_temp = eig_vec_temp(:, 1:num_modes); 
    eig_vec = zeros(num_vertices, num_modes); 
    eig_vec(cortex_ind,:) = eig_vec_temp(cortex_ind,1:num_modes); 

    % reconstruct dataand calculate objective
    recon_beta_task = zeros(num_modes, num_modes, num_subjects); 
    for subj = 1:num_subjects
        for mode = 1:num_modes
            basis = eig_vec(cortex_ind, 1:mode); 
            recon_beta_task(1:mode, mode, subj) = calc_eigendecomposition(data_recon(cortex_ind, subj), basis, 'matrix');
        end
    end
    
    data_plot = zeros(num_vertices, num_subjects);
    minimize = zeros(1, num_subjects);
    MSE = zeros(num_vertices, num_subjects);
    N = num_modes; 
    recon_beta_squeeze = squeeze(recon_beta_task); 

    for subj = 1:num_subjects
        disp(['Size eig_vec: ', mat2str(size(eig_vec))]);
        disp(['Size recon_beta_task: ', mat2str(size(recon_beta_task))]);
        disp(['Size data_plot: ', mat2str(size(data_plot))]);

        data_plot(:,subj) = eig_vec(:,1:N) * (recon_beta_squeeze(1:N,N, subj));

        minimize(:,subj) = nanmean(data_recon(:,subj) - data_plot(:,subj));

        for i = 1:num_vertices
            MSE(i, subj) = nanmean((data_recon(i,subj) - data_plot(i,subj))).^2; 
        end
        total_MSE(subj) = sum(MSE(:, subj));
    end
    % Ensure obj is a scalar
    disp(['Total MSE: ', num2str(total_MSE)]);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% demo_eigenmode_analysis.m
%%%
%%% MATLAB script to demonstrate how to use surface eigenmodes to analyze 
%%% fMRI data. In particular, the script demonstrates how to
%%% (1) reconstruct a task fMRI spatial map,
%%% (2) reconstruct a resting-state fMRI spatiotemporal map and functional
%%%     connectivity (FC) matrix, and
%%% (3) calculate the eigenmode-based power spectral content of a spatial map
%%%
%%% NOTE 1: The script can also be used to analyze fMRI data using other
%%%         types of surface eigenmodes (e.g., connectome eigenmodes). 
%%%         Just change the eigenmodes variable below. However, make sure
%%%         that the variable is an array of size
%%%         [number of vertices x number of modes]. 
%%% NOTE 2: Current demo uses 50 modes. For a proper analysis, we advise 
%%%         using between 100 to 200 modes. 200 template geometric 
%%%         eigenmodes are provided in data/template_eigenmodes.
%%%
%%% Original: James Pang, Monash University, 2022
%%%
%%% Adjusted: Jill Bay, MPI Leipzig, 2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load relevant repository MATLAB functions

addpath(genpath('functions_matlab'));

%% Load surface files for visualization

surface_interest = 'fsLR_32k';
hemisphere = 'lh';
mesh_interest = 'midthickness';

[vertices, faces] = read_vtk(sprintf('data/template_surfaces_volumes/%s_%s-%s.vtk', surface_interest, mesh_interest, hemisphere));
surface_midthickness.vertices = vertices';
surface_midthickness.faces = faces';

% Load cortex mask
cortex = dlmread(sprintf('data/template_surfaces_volumes/%s_cortex-%s_mask.txt', surface_interest, hemisphere));
cortex_ind = find(cortex);

%% Reconstruct a single-subject task fMRI spatial map

hemisphere = 'lh';
num_modes = 200;

% =========================================================================
%                    Load eigenmodes and empirical data           
% =========================================================================

% Select the relevant eigenmodes: geometric, connectome, EDR, or hybrid. 

% % Connectome density-matched: eigenmodes:
% eigenmodes = load(sprintf('data/examples/connectome_density-matched_eigenmodes-lh_200.mat')); 
% eigenmodes = eigenmodes.eig_vec; 

% Geometric eigenmodes: 
eigenmodes = load(sprintf('data/template_eigenmodes/fsLR_32k_midthickness-lh_emode_200.txt', hemisphere));

% % Connectome: 
% eigenmodes = load('data/examples/connectome_eigenmodes-lh_200.mat'); 
% eigenmodes = eigenmodes.eig_vec; 

% Hybrid eigenmodes:
%  eigenmodes = load(sprintf('data/eigenmodes.mat')); 
%  eigenmodes = eigenmodes.eig_vec; 

% EDR eigenmodes:
%  eigenmodes = load(sprintf('data/examples/synthetic_EDRconnectome_eigenmodes-lh_200.mat')); 
%  eigenmodes = eigenmodes.eig_vec; 

% Load data
data = load(sprintf('data/empirical/S255_tfMRI_ALLTASKS_raw_%s', hemisphere));
data_to_reconstruct = data.zstat; 
%task_data = data_to_reconstruct.motor_cue_avg; 

field_names = fieldnames(data_to_reconstruct); 
num_tasks = numel(field_names); 

% =========================================================================
% Calculate reconstruction beta coefficients for each subject
% =========================================================================

% These following structs will contain the weights (recon_beta), and the
% reconstruction accuracy on the parcellation (recon_corr_parc) of all the
% 47 tasks.
recon_beta_task = struct(); 
recon_corr_parc = struct(); 

for task_id = 1:num_tasks
    task_name = field_names{task_id}; 
    task_data = data_to_reconstruct.(task_name); 
    num_subjects = size(task_data, 2);

    recon_beta_task = zeros(num_modes, num_modes, num_subjects); 
    
    for subj = 1:num_subjects
        for mode = 1:num_modes
            basis = eigenmodes(cortex_ind, 1:mode); 
            recon_beta_task(1:mode, mode, subj) = calc_eigendecomposition(task_data(cortex_ind, subj), basis, 'matrix'); 
        end 
    end 
    
    recon_beta_struct.(task_name) = recon_beta_task; 

    % =========================================================================
    %     Calculate reconstruction accuracy for each subject   
    % =========================================================================
    recon_corr_vertex = zeros(num_subjects, num_modes);               
    recon_corr_parc = zeros(num_subjects, num_modes);    

    % At parcellated level
    parc_name = 'Glasser360';
    parc = dlmread(sprintf('data/parcellations/fsLR_32k_%s-%s.txt', parc_name, hemisphere));
    
    for subj = 1:num_subjects
        for mode = 1:num_modes
            recon_temp = eigenmodes(cortex_ind, 1:mode)*recon_beta_task(1:mode, mode, subj);
            recon_corr_vertex(subj, mode) = corr(task_data(cortex_ind, subj), recon_temp);
    
            recon_temp_parc = eigenmodes(:, 1:mode)*recon_beta_task(1:mode, mode, subj);
            recon_corr_parc(subj, mode) = corr(calc_parcellate(parc, task_data(:, subj)), calc_parcellate(parc, recon_temp_parc));
        end
    end
    
    % based on which eigenmodes you select, you change the name of the
    % following variable:
    task_recon_corr_geometric.(task_name) = recon_corr_parc; 
end

% =========================================================================
%                      Some visualizations of results                      
% =========================================================================
N = size(eigenmodes,2);
surface_to_plot = surface_midthickness;

% Data_to_plot is the data reconstructed for every single subject.
data_to_plot = zeros(size(task_data, 1), num_subjects); 
for subj = 1:num_subjects
    data_to_plot(:,subj) = eigenmodes(:, 1:N)*recon_beta_task(1:N,N, subj);
end

medial_wall = find(cortex==0);
with_medial = 1;

% Compute the average reconstructed data over all subjects
% (task_map_recon):
task_map_emp = nanmean(task_data, 2);
task_map_recon = eigenmodes(:, 1:N) * calc_eigendecomposition(task_map_emp(cortex_ind), eigenmodes(cortex_ind, 1:N), 'matrix');
medial_wall = find(cortex==0);
with_medial = 1;

% Figure that displays the reconstructed data as a surface map of the
% cortex.
fig = draw_surface_bluewhitered_dull(surface_to_plot, task_map_recon, hemisphere, medial_wall, with_medial);
fig.Name = sprintf('tfMRI reconstruction - surface map using %i modes (average over subjects)', N);

% Figure that displays the reconstruction accuracy for a specific task
% (task_name): 
figure(1); 
plot(mean(recon_corr_parc_struct.(task_name)), LineWidth=2);
title('Reconstruction accuracy', task_name); 
xlabel('Number of modes (N)'); 
ylabel('Reconstruction accuracy (*100%)');
grid on;

% Figure that displays the reconstruction accuracy vs number of modes at 
% vertex and parcellated levels. We only use the parcellated levels. 
figure('Name', 'tfMRI reconstruction - accuracy');
hold on;
plot(1:num_modes, recon_corr_vertex, 'k-', 'linewidth', 2, 'displayname', 'vertex')
plot(1:num_modes, recon_corr_parc, 'b-', 'linewidth', 2, 'displayname', 'parcellated')
hold off;
leg = legend('fontsize', 12, 'location', 'southeast', 'box', 'off');
set(gca, 'fontsize', 10, 'ticklength', [0.02 0.02], 'xlim', [1 num_modes], 'ylim', [0 1])
xlabel('number of modes', 'fontsize', 12)
ylabel('reconstruction accuracy', 'fontsize', 12)
grid on; 


%% Reconstruct a single-subject resting-state fMRI spatiotemporal map and FC matrix

hemisphere = 'lh';
num_modes = 200;

% Load HCP subject list
subject_list = importdata('subject_list_HCP.txt'); 

num_subjects = length(subject_list);

% At parcellated level
parc_name = 'Glasser360'; 
parc = dlmread(sprintf('data/parcellations/fsLR_32k_%s-%s.txt', parc_name, hemisphere)); 
num_parcels = length(unique(parc(parc>0))); 

FC_matrices = zeros(num_parcels, num_parcels, num_subjects); 
FC_emp_all = zeros(num_parcels, num_parcels, num_subjects); 

% =========================================================================
%                    Load eigenmodes and empirical data                    
% =========================================================================

% % Connectome density-matched: eigenmodes:
% eigenmodes = load(sprintf('data/examples/connectome_density-matched_eigenmodes-lh_200.mat')); 
% eigenmodes = eigenmodes.eig_vec; 

% Connectome  
% eigenmodes = load(sprintf('data/examples/connectome_eigenmodes-lh_200.mat')); 
% eigenmodes = eigenmodes.eig_vec; 

% Geometric eigenmodes: 
% eigenmodes = load(sprintf('data/template_eigenmodes/fsLR_32k_midthickness-lh_emode_200.txt'));

% EDR eigenmodes:
 eigenmodes = load(sprintf('data/examples/synthetic_EDRconnectome_eigenmodes-lh_200.mat')); 
 eigenmodes = eigenmodes.eig_vec; 

% Hybrid eigenmodes 50/50: 
% eigenmodes = load(sprintf('eig_vec_combined.mat')); 
% eigenmodes = eigenmodes.eig_vec; 


MSE_array = zeros(num_subjects, 1); 
likelihood_array = zeros(num_subjects, 1); 
BIC_array = zeros(num_subjects, 1); 

recon_corr_parc = zeros(num_subjects, num_modes); 

% Load rfMRI time series data for every subject in the subject_list_HCP
for subj = 1:num_subjects
    subj_id = subject_list(subj); 
    data = sprintf('%d_rfMRI_REST1_LR_Atlas.dtseries.nii', subj_id); 
    
    data_dtseries = ft_read_cifti(data); 
    if strcmpi(hemisphere, 'lh')
        data_resting = data_dtseries.dtseries(data_dtseries.brainstructure==1,:); 
    elseif strcmpi(hemisphere, 'rh')
        data_resting = data_dtseries.dtseries(data_dtseries.brainstructrue==2,:); 
    end

    T = size(data_resting, 2); 
    clear data_dtseries; 

    % =====================================================================
    % Calculate reconstruction beta coefficients using 1 to num_modes eigenmodes
    % =====================================================================

    recon_beta = zeros(num_modes, T, num_modes); 

    for mode = 1:num_modes 
        basis = eigenmodes(cortex_ind, 1:mode); 
        recon_beta(1:mode, :, mode) = calc_eigendecomposition(data_resting(cortex_ind,:), basis, 'matrix'); 
    end

    % =====================================================================
    % Calculate reconstruction accuracy using 1 to num_modes eigenmodes
    % =====================================================================

    % Extract upper triangle indices; 
    triu_ind = calc_triu_ind(zeros(num_parcels, num_parcels)); 

    % Calculate empirical FC 
    data_parc_emp = calc_parcellate(parc, data_resting); 
    data_parc_emp = calc_normalize_timeseries(data_parc_emp'); 
    data_parc_emp(isnan(data_parc_emp)) = 0; 

    FC_emp = data_parc_emp'*data_parc_emp; 
    FC_emp = FC_emp/T; 
    FCvec_emp = FC_emp(triu_ind); 

    % Calculate reconstructed FC and accuracy
    FCvec_recon = zeros(length(triu_ind), num_modes); 

    for mode = 1:num_modes
        recon_temp = eigenmodes(:, 1:mode) * squeeze(recon_beta(1:mode, :, mode)); 

        data_parc_recon = calc_parcellate(parc, recon_temp); 
        data_parc_recon = calc_normalize_timeseries(data_parc_recon'); 
        data_parc_recon(isnan(data_parc_recon)) = 0; 

        FC_recon_temp = data_parc_recon'*data_parc_recon; 
        FC_recon_temp = FC_recon_temp/T; 

        FCvec_recon(:,mode) = FC_recon_temp(triu_ind); 

        recon_corr_parc(subj, mode) = corr(FCvec_emp, FCvec_recon(:, mode)); 
    end

    N = num_modes; 

    % Reconstructed FC using N = num_modes modes 
    FC_recon = zeros(num_parcels, num_parcels); 
    FC_recon(triu_ind) = FCvec_recon(:,N); 
    FC_recon = FC_recon + FC_recon'; 
    FC_recon(1:(num_parcels+1):num_parcels^2) =1; 

    FC_matrices(:,:,subj) = FC_recon;
    FC_emp_all(:,:,subj) = FC_emp; 
    
    % based on which eigenmodes you select, you change the name of the
    % following variable:
    task_recon_corr_geometric = recon_corr_parc; 
end

% Average across subjects 
average_FC = mean(FC_matrices, 3); 
FC_emp_average = mean(FC_emp_all, 3); 


% =========================================================================
%                          Shape FC-matrix                          
% =========================================================================

% In order to get the correct FC_combined in the end, 

modesets = 3;
FC_modesets = zeros(num_parcels, num_parcels, modesets); 
FC_modesets(:,:,1) = average_FC; % for num_modes = 10; 
FC_modesets(:,:,2) = average_FC; % for num_modes = 100; 
FC_modesets(:,:,3) = average_FC; % for num_modes = 200; 

FC_combined = zeros(num_parcels, num_parcels, modesets); 
for i = 1:modesets
    lower_tri_indices = tril(true(size(FC_modesets, 1)), -1); 
    FC_combined(:,:,i) = average_FC .* ~lower_tri_indices + FC_emp_average .* lower_tri_indices;
end

FC_low = zeros(187,187,3); 
FC_up = zeros(187,187,3);
FC = zeros(187,187,3);

% Define the starting indices for the lower triangle of the original matrix
% to be copied into the new matrix. We start at index 1,1
start_row = 8;
start_col = 1;

% Copy the lower triangular part of the original matrix into the new matrix
for i = 1:size(FC_combined, 3) % Loop through each slice if it's a 3D matrix
    % Use the tril function to extract the lower triangular part including the diagonal (k=0)
    lower_triangular_part = tril(FC_combined(:,:,i), 0);
    
    % Paste the lower triangular part into the corresponding section of the new matrix
    FC_low(start_row:(start_row+size(FC_combined,1)-1), ...
               start_col:(start_col+size(FC_combined,2)-1), ...
               i) = lower_triangular_part;
end

% Define the ending indices for the upper triangle of the original matrix
% to be copied into the new matrix. Since we want to place it in the upper
% part of the new matrix, we calculate the end_row and end_col based on the size difference
start_row = 1;
start_col = 8;


% Copy the upper triangular part of the original matrix into the new matrix
for i = 1:size(FC_combined, 3) % Loop through each slice if it's a 3D matrix
    % Use the triu function to extract the upper triangular part including the diagonal (k=0)
    upper_triangular_part = triu(FC_combined(:,:,i), 0);
    
    % Paste the upper triangular part into the corresponding section of the new matrix
     FC_up(start_row:(start_row+size(FC_combined,1)-1), ...
               start_col:(start_col+size(FC_combined,2)-1), ...
               i) = upper_triangular_part;
end

for i = 1:size(FC_combined,3) 
    FC(:,:,i) = FC_up(:,:,i) + FC_low(:,:,i); 
end

FC_combined = FC; 

% =========================================================================
%                      Some visualizations of results                      
% =========================================================================

% Reconstruction accuracy vs number of modes at parcellated level
figure('Name', 'rfMRI reconstruction - accuracy');
hold on;
plot(1:num_modes, recon_corr_parc, 'b-', 'linewidth', 2)
hold off;
set(gca, 'fontsize', 10, 'ticklength', [0.02 0.02], 'xlim', [1 num_modes], 'ylim', [0 1])
xlabel('number of modes', 'fontsize', 12)
ylabel('reconstruction accuracy', 'fontsize', 12)

% FC matrix of different modesets. 
N = 3; % set to 1=10 modes, 2=100 modes, 3 = 200 modes
figure('Name', sprintf('rfMRI reconstruction - FC matrix using %i modes', N));
imagesc(FC_combined(:,:,N)) 
caxis([-1 1])
colormap(bluewhitered)
cbar = colorbar;
set(gca, 'fontsize', 10, 'ticklength', [0.02 0.02])
xlabel('region', 'fontsize', 12)
ylabel('region', 'fontsize', 12)
ylabel(cbar, 'FC', 'fontsize', 12)
axis image


%% Calculate modal power spectral content of spatial maps
hemisphere = 'lh';
num_modes = 200;

% =========================================================================
%                   Load eigenmodes and empirical data
% =========================================================================

% Load 50 fsLR_32k template midthickness surface eigenmodes
% eigenmodes = dlmread(sprintf('data/examples/fsLR_32k_midthickness-%s_emode_%i.txt', hemisphere, num_modes));

% Replace above line with the one below and make num_modes = 200 if using the 200 modes provided at data/template_eigenmodes
eigenmodes = dlmread(sprintf('data/template_eigenmodes/fsLR_32k_midthickness-%s_emode_%i.txt', hemisphere, num_modes));

data = load('data/empirical/S255_tfMRI_ALLTASKS_raw_lh.mat'); 
data_to_reconstruct = data.zstat;

task_names = fieldnames(data_to_reconstruct);
num_tasks = length(task_names);

spectrum_HCP = zeros(num_modes, num_tasks);

for task_idx = 1:num_tasks

    % Extract the task data
    task_data = data_to_reconstruct.(task_names{task_idx});
    
    % Average
    group_avg_data = nanmean(task_data, 2);
    
    % Calculate reconstruction beta coefficients
    basis = eigenmodes(cortex_ind, 1:num_modes);    
    recon_beta = calc_eigendecomposition(group_avg_data(cortex_ind), basis, 'matrix');
    
    % Calculate the modal power spectrum
    [~, normalized_power_spectrum] = calc_power_spectrum(recon_beta);
    
    % Store the results in spectrum_HCP
    spectrum_HCP(:, task_idx) = normalized_power_spectrum;
end

% Normalized power spectrum
figure('Name', 'rfMRI reconstruction - accuracy');
bar(1:num_modes, spectrum_HCP(:,1))
set(gca, 'fontsize', 10, 'ticklength', [0.02 0.02], 'xlim', [2 num_modes], 'yscale', 'log')
xlabel('mode', 'fontsize', 12)
ylabel('normalized power (log scale)', 'fontsize', 12);


%% Reconstruct contributions of long- and short-wavelength modes
hemisphere = 'lh';
num_modes = 200;

% =========================================================================
%                    Load eigenmodes and empirical data                    
% =========================================================================

% Load 50 fsLR_32k template midthickness surface eigenmodes
%eigenmodes = dlmread(sprintf('data/examples/fsLR_32k_midthickness-%s_emode_%i.txt', hemisphere, num_modes));

% Replace above line with the one below and make num_modes = 200 if using the 200 modes provided at data/template_eigenmodes
eigenmodes = load(sprintf('data/template_eigenmodes/fsLR_32k_midthickness-lh_emode_200.txt', hemisphere));

% Load example single-subject tfMRI z-stat data
data = load(sprintf('data/empirical/S255_tfMRI_ALLTASKS_raw_%s', hemisphere));
data_to_reconstruct = data.zstat;

% Select each of the 7 key tasks:
data_task = data_to_reconstruct.gambling_punish_reward; 

num_subjects = size(data_task,2); 


% =========================================================================
% Calculate reconstruction beta coefficients for each subject
% =========================================================================

parc_name = 'Glasser360';
parc = dlmread(sprintf('data/parcellations/fsLR_32k_%s-%s.txt', parc_name, hemisphere));

for mode = 1:num_modes
   % long-wavelength mode removal (removing modes 1, 1-2, 1-3, ...)
   excluded_modes_long = 1:mode;
   included_modes_long = setdiff(1:num_modes, excluded_modes_long);
   
   % short-wavelength mode removal (removing modes 200, 200-199, 200-198, ...)
   excluded_modes_short = (num_modes-mode+1):num_modes;
   included_modes_short = setdiff(1:num_modes, excluded_modes_short);
   
   accuracy_long_subjects = zeros(1,num_subjects);
   accuracy_short_subjects = zeros(1,num_subjects);
   
   for subj=1:num_subjects
       % reconstruction using the modes that are not excluded
       recon_data_long = eigenmodes(:, included_modes_long) * calc_eigendecomposition(data_task(cortex_ind, subj), eigenmodes(cortex_ind, included_modes_long),'matrix');
       recon_data_short = eigenmodes(:, included_modes_short) * calc_eigendecomposition(data_task(cortex_ind, subj), eigenmodes(cortex_ind, included_modes_short), 'matrix');
       
       % compute accuracy as correlation between empirical and reconstructed data
        
        % At parcellated level
       accuracy_long_subjects(subj) = corr(calc_parcellate(parc, data_task(:, subj)), calc_parcellate(parc, recon_data_long), 'rows', 'complete');
       accuracy_short_subjects(subj) = corr(calc_parcellate(parc, data_task(:, subj)), calc_parcellate(parc, recon_data_short), 'rows', 'complete');
   end

    task_recon_remove_highf_geometric(mode) = nanmean(accuracy_long_subjects);
    task_recon_remove_lowf_geometric(mode) = nanmean(accuracy_short_subjects);
end
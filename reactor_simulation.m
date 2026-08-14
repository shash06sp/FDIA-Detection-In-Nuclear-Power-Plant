clear; clc;
disp('Initializing High-Speed 5D Cyber-Physical Simulation...');

% --- 1. PHYSICAL PARAMETERS ---
beta = 0.00656; Lambda = 0.0001; lam = 0.08; alpha_T = -0.015; alpha_X = -0.02;     
gamma_f = 0.1; tau_c = 0.05; k_steam = 0.02; gamma_x = 0.012; lam_x = 0.008;       
k_valve = 0.05; tau_v = 0.1; dt = 0.1;

A = [-beta/Lambda, lam,  alpha_T/Lambda, alpha_X/Lambda, 0;
      beta/Lambda, -lam,  0,              0,              0;
      gamma_f,      0,   -tau_c,          0,             -k_steam;
      gamma_x,      0,    0,             -lam_x,          0;
      0,            0,    k_valve,        0,             -tau_v];
B = [1.0; 0.0; 0.0; 0.0; 0.0]; C = eye(5); D = zeros(5, 1);

sys_continuous = ss(A, B, C, D);
sys_discrete = c2d(sys_continuous, dt, 'zoh');
Ad = sys_discrete.A; Bd = sys_discrete.B;

Q_cost = diag([10, 1, 50, 1, 100]); R_cost = 1.0;            
[K, ~, ~] = dlqr(Ad, Bd, Q_cost, R_cost);

% --- 2. FAST NOISE GENERATION (CHOLESKY) ---
Q_noise = diag([0.05, 0.01, 0.1, 0.01, 0.05]); 
R_noise = diag([0.1, 0.1, 0.5, 0.1, 0.2]);

% Compute lower triangular Cholesky factors (L * L' = Covariance)
L_Q = chol(Q_noise, 'lower'); 
L_R = chol(R_noise, 'lower');

% --- 3. PRE-ALLOCATION FOR PARALLEL PROCESSING ---
n_steps = 600;  
n_simulations = 30000; 
rows_per_sim = floor(n_steps / 10); 
dataset_3D = zeros(n_simulations, rows_per_sim, 18); 
disp(['Starting Parallel Pool for ', num2str(n_simulations), ' simulations...']);
tic;

% --- 4. PARALLEL MONTE CARLO LOOP ---
parfor sim_id = 1:n_simulations 
    
    x = zeros(5, n_steps);
    z = zeros(5, n_steps);
    local_data = zeros(rows_per_sim, 18);
    local_idx = 1;
    
    x(:, 1) = L_Q * randn(5, 1);
    
    rand_val = rand();
    if rand_val < 0.4
        scenario = 0; 
    elseif rand_val < 0.7
        scenario = 1; 
    else
        scenario = 2; 
    end
    
    attack_start = randi([100, 300]); 
    process_noise = (L_Q * randn(5, n_steps)) * sqrt(dt);
    sensor_noise = (L_R * randn(5, n_steps));
    
    for k = 2:n_steps
        grid_demand = 2.0 * sin(2 * pi * (k * dt) / 200); 
        x_ref = [grid_demand; 0; 0; 0; 0];
        u = -K * (x(:, k-1) - x_ref);
        u = max(min(u, 10.0), -10.0);
        x(:, k) = Ad * x(:, k-1) + Bd * u + process_noise(:, k);
        y = x(:, k) + sensor_noise(:, k);
        y = max(min(y, 100.0), -100.0);
        
        if k >= attack_start
            if scenario == 1
                y(3) = y(3) - 20.0; 
            elseif scenario == 2
                time_in_attack = k - attack_start;
                stealth_drift = 0.0002 * (time_in_attack^2); 
                y(3) = y(3) - stealth_drift;  
                y(1) = y(1) + stealth_drift * (alpha_T/gamma_f); 
            end
        end
        
        z(:, k) = y - (Ad * x(:, k-1) + Bd * u);
        
        if mod(k, 10) == 0
            local_data(local_idx, :) = [(sim_id-1), k*dt, scenario, ...
                x(1,k), x(2,k), x(3,k), x(4,k), x(5,k), ...
                y(1), y(2), y(3), y(4), y(5), ...
                z(1,k), z(2,k), z(3,k), z(4,k), z(5,k)];
            local_idx = local_idx + 1;
        end
    end
    dataset_3D(sim_id, :, :) = local_data;
end
toc; 

% --- 5. TENSOR FLATTENING & EXPORT ---
disp('Flattening 3D Tensor to 2D Matrix...');
dataset_2D = reshape(permute(dataset_3D, [2, 1, 3]), [], 18);

disp('Writing to CSV...');
varNames = {'Sim_ID', 'Time', 'Label', ...
           'True_Flux', 'True_Prec', 'True_Temp', 'True_Xenon', 'True_Valve', ...
           'Sens_Flux', 'Sens_Prec', 'Sens_Temp', 'Sens_Xenon', 'Sens_Valve', ...
           'Res_Flux', 'Res_Prec', 'Res_Temp', 'Res_Xenon', 'Res_Valve'};
outputTable = array2table(dataset_2D, 'VariableNames', varNames);
parquetwrite('cps_training_data_30k.parquet', outputTable);
disp('SUCCESS: Accelerated Data generation complete.');
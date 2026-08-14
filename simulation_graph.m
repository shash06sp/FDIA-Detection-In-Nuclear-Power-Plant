clear; clc; close all;
disp('Generating Publication-Quality 5D Plant Simulation Graph...');

% --- 1. PHYSICAL PARAMETERS & MATRICES ---
beta = 0.00656; Lambda = 0.0001; lam = 0.08; alpha_T = -0.015; alpha_X = -0.02;     
gamma_f = 0.1; tau_c = 0.05; k_steam = 0.02; gamma_x = 0.012; lam_x = 0.008;       
k_valve = 0.05; tau_v = 0.1; dt = 0.1; n_steps = 600; time = (0:n_steps-1) * dt;

A = [-beta/Lambda, lam,  alpha_T/Lambda, alpha_X/Lambda, 0;
    beta/Lambda, -lam,  0,              0,              0;
    gamma_f,      0,   -tau_c,          0,             -k_steam;
    gamma_x,      0,    0,             -lam_x,          0;
    0,            0,    k_valve,        0,             -tau_v];
B = [1.0; 0.0; 0.0; 0.0; 0.0]; C = eye(5); D = zeros(5, 1);
sys_discrete = c2d(ss(A, B, C, D), dt, 'zoh');
Ad = sys_discrete.A; Bd = sys_discrete.B;
[K, ~, ~] = dlqr(Ad, Bd, diag([10, 1, 50, 1, 100]), 1.0);

% --- 2. RUN A SINGLE STEALTH ATTACK SIMULATION ---
x = zeros(5, n_steps); y = zeros(5, n_steps);
attack_start = 200; 

for k = 2:n_steps
    grid_demand = 2.0 * sin(2 * pi * (k * dt) / 200); 
    u = max(min(-K * (x(:, k-1) - [grid_demand; 0; 0; 0; 0]), 10.0), -10.0);
    x(:, k) = Ad * x(:, k-1) + Bd * u + (randn(5,1)*0.01);
    y(:, k) = x(:, k) + (randn(5,1)*0.05);

    if k >= attack_start
        stealth_drift = 0.002 * (k - attack_start); 
        y(3, k) = y(3, k) - stealth_drift;  
        y(1, k) = y(1, k) + stealth_drift * (alpha_T/gamma_f); 
    end
end

% --- 3. ACADEMIC GRAPHING (TILED LAYOUT) ---
fig = figure('Position', [100, 100, 900, 800], 'Color', 'w');
t = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Graph 1: Core Temperature
ax1 = nexttile;
plot(time, x(3,:), 'Color', [0.6350 0.0780 0.1840], 'LineWidth', 2.5); hold on;
plot(time, y(3,:), '--', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2.5);
xline(attack_start*dt, 'k:', 'LineWidth', 2);
ylabel('Temperature (\delta T_f)', 'FontWeight', 'bold');
title('Cross-Channel Thermodynamic Divergence Under FDIA', 'FontSize', 14);
legend('True Physical State', 'Spoofed Sensor (SCADA)', 'Location', 'southwest');
grid on; set(gca, 'FontSize', 11);

% Graph 2: Neutron Flux
ax2 = nexttile;
plot(time, x(1,:), 'Color', [0.0 0.4470 0.7410], 'LineWidth', 2.5); hold on;
plot(time, y(1,:), '--', 'Color', [0.3010 0.7450 0.9330], 'LineWidth', 2.5);
xline(attack_start*dt, 'k:', 'LineWidth', 2);
ylabel('Neutron Flux (\delta n)', 'FontWeight', 'bold');
grid on; set(gca, 'FontSize', 11);

% Graph 3: Xenon-135 Poisoning
ax3 = nexttile;
plot(time, x(4,:), 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 2.5); hold on;
xline(attack_start*dt, 'k:', 'LineWidth', 2);
ylabel('Xenon-135 (\delta X)', 'FontWeight', 'bold');
xlabel('Operational Time (Seconds)', 'FontWeight', 'bold');
grid on; set(gca, 'FontSize', 11);

linkaxes([ax1, ax2, ax3], 'x');
xlim([0, n_steps*dt]);

exportgraphics(fig, 'matlab_plant_simulation.pdf', 'ContentType', 'vector');
disp('SUCCESS: Graph saved as matlab_plant_simulation.pdf');
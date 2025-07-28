clear all
 g = 0.007;       % Growth rate of T
 g_E = 0;        % Growth rate of E
 k_on = 0.0003;    % Binding rate
 k_off = 0.3;  % Dissociation rate
 k_kill = 0.007; % Killing rate
TStart = 0;
TFinal = 100;
E0_range = linspace(1,10000,50);  % Renamed to avoid confusion with initial condition
T0_range = linspace(1,10000,50);  % Renamed to avoid confusion with initial condition
n = 200;
th = 10^(-16);
th2 = 10;
no_of_interval = 20;
% tfinal = 50000;

% Initialize F matrix
F = zeros(length(E0_range), length(T0_range));

for i = 1:length(E0_range)
    for j = 1:length(T0_range)
        % Initial conditions: [T, E, C]
        u0 = [T0_range(j), E0_range(i), 0];
        
        % Run simulation
        [t, T, E, C] = tumor(TStart, TFinal, u0, g, g_E, k_on, k_off, k_kill);
        
        % Control simulation (no E cells)
        [t0, T0, E0_ctrl, C0] = tumor(TStart, TFinal, [T0_range(j) 0 0], g, g_E, k_on, k_off, k_kill);
        
        % Interpolate control to match time points
        T0_interp = interp1(t0, T0, t);
        
        % Calculate the ratio at the final time point
        F(i,j) = T(end) / T0_interp(end);
    end
end

figure(1)
surf(E0_range, T0_range, F')
% title('k_{on}=0.1')
colorbar;
set(gca,'YDir', 'normal')
xlabel('E0','Fontsize',18)
ylabel('T0','Fontsize',18)
zlabel('F', 'FontSize',18)
grid on
grid minor

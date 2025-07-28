function tumor_analysis()
    % Parameters
    g = 0.01;       % Growth rate of T
    g_E = 0;        % Growth rate of E
    k_on = 0.01;    % Binding rate
    k_off = 0.005;  % Dissociation rate
    k_kill = 0.002; % Killing rate 
    
    TStart = 0;
    TFinal = 100;
    
    % Case 1: With effector (E0 = 100)
    u0 = [100 100 0];
    [t, T, E, C] = tumor(TStart, TFinal, u0, g, g_E, k_on, k_off, k_kill);
    
    % Case 2: Without effector (E0 = 0)
    [t0, T0, E0, C0] = tumor(TStart, TFinal, [100 0 0], g, g_E, k_on, k_off, k_kill);
    
    % Ensure both simulations have the same time points for comparison
    % (Interpolate if necessary, but here we assume ode45 uses adaptive steps)
    % For simplicity, we use the time points from the first simulation
    T0_interp = interp1(t0, T0, t); % Interpolate T0 to match t
    
    % Calculate F(t) = T(t;E0)/T(t;0)
    F = T ./ T0_interp;
    
    % Plot dynamics
    figure(1)
%     subplot(2,1,1)
    plot(t, T, 'r', t, E, 'b', t, C, 'k', 'LineWidth', 2)
    legend('Target', 'Effector', 'Complex', 'Location', 'best')
    xlabel('Time')
    ylabel('T,E,C')
    title('With Effector')
    set(gca, 'FontSize', 12)
    grid on
    
     figure(2)
%     subplot(2,1,2)
    plot(t0, T0, 'r', t0, E0, 'b', t0, C0, 'k', 'LineWidth', 2)
    %plot(t0, T0, 'r--', 'LineWidth', 2)
    legend('Target', 'Effector', 'Complex', 'Location', 'best')
    xlabel('Time')
    ylabel('Tumor Population')
    title('Without Effector E_0=0')
    set(gca, 'FontSize', 12)
    grid on
    
    % Plot F(t) over time
    figure(3)
    plot(t, F, 'm-', 'LineWidth', 2)
    xlabel('Time')
    ylabel('F(t) = T(t;E_0)/T(t;0)')
    title('Evolution of F Ratio Over Time')
    grid on
    set(gca, 'FontSize', 12)
    
    % Display final F value
    fprintf('Final F ratio at t = %d: F = %.4f\n', TFinal, F(end));
end


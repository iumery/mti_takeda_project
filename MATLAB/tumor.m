function [t, T, E,C] = tumor(TStart, TFinal, u0, g, g_E, k_on, k_off, k_kill)

[t, u] = ode45(@f, [TStart TFinal], u0);
% u is a matrix with rows = time, columns = variables
% column 1 = T; column 2 = E, column 3=C

T = u(:, 1);
E = u(:, 2);
C = u(:,3);

function dudt = f(t, u)
        T =  u(1); E = u(2); C=u(3);
        dudt = [g * T - k_on * E * T + k_off * C; -k_on * E * T + k_off * C + k_kill * C + g_E * E; k_on * E * T - k_off * C - k_kill * C];
end
         
end
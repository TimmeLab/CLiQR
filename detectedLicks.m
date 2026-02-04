% detectedLicks.m
% This script will create graphs on a per-animal basis, showing the
% full trace of capacitive data with red dots overlaid where licks
% were detected.

function detectedLicks(animal_num, capTime, capValues, lickTimes)
% Ensure that you've loaded the results from lickDetector.m into the
% workspace or these vars will be uninitialized.
times = capTime(:, animal_num);
values = capValues(:, animal_num);
lick_times = lickTimes(animal_num, :)';

[N, C] = size(times);

for n = 1:N
    t = times{n};
    v = values{n};
    l = lick_times{n};
    % Find indices of events in the time vector
    % (robust to floating-point issues)
    [tf, lick_idx] = ismembertol(l, t, 1e-9);
    
    % Warn if any events were not matched
    if any(~tf)
        warning('%d event times were not found in the time vector.', sum(~tf));
    end
    
    lick_idx = lick_idx(tf);
    
    % Plot main signal
    figure;
    plot(t, v, 'b-');
    hold on;
    
    % Overlay red dots at event times
    plot(t(lick_idx), v(lick_idx), 'r.', 'MarkerSize', 14);
    
    xlabel('Time (s)');
    ylabel('Capacitance (pF)');
    title('Detected Licks');
    grid on;
    hold off;
end
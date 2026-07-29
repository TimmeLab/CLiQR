% scripts/matlab_parity_refs.m
% Dump (input, softmax-output) reference pairs from the trained MATLAB nets so the PyTorch
% port can be validated numerically. Uses fixed seeds for reproducibility.
% NOTE: MATLAB's run() temporarily cd's into this script's own folder before executing it,
% so re-anchor to the project root (parent of scripts/) before using any relative paths.
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
cd(projectRoot);
S = load(fullfile('ML Detection MATLAB Code', 'lickNets.mat'));
rng(0);
K = 8;
bout_in = single(randn(K, 300));
point_in = single(randn(K, 21));
bout_out = zeros(K, 2);
point_out = zeros(K, 2);
for i = 1:K
    xb = reshape(bout_in(i, :), [1 300 1 1]);
    yb = predict(S.netBout, xb);      % softmax probabilities
    bout_out(i, :) = yb(:)';
    xp = reshape(point_in(i, :), [1 21 1 1]);
    yp = predict(S.netPoint, xp);
    point_out(i, :) = yp(:)';
end
save(fullfile('ml_detection', 'checkpoints', 'parity_refs.mat'), ...
     'bout_in', 'bout_out', 'point_in', 'point_out', '-v7');
disp('PARITY_REFS_DONE');

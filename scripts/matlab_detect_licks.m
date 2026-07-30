function matlab_detect_licks(inMat, netFile, outMat)
% matlab_detect_licks  Run the ORIGINAL MATLAB lick-detection cascade on one trace.
%
% This is the ground-truth oracle for validating the Python/PyTorch port. It calls the
% unmodified detectLicksFromRaw (netBout + netPoint cascade) on a single capacitance trace
% and writes the detected lick times to disk so Python can compare against them.
%
% INPUT:
%   inMat   : .mat file containing a variable `rawData`, an [nSamples x 2] matrix whose first
%             column is time (seconds) and second column is raw capacitance. This is exactly
%             the input shape detectLicksFromRaw expects.
%   netFile : path to lickNets.mat (the trained netBout/netPoint/meta).
%   outMat  : path to write results; will contain `lickTimes` (column vector, seconds).
%
% USAGE (headless, single line — MATLAB -batch dislikes newlines in the command string):
%   matlab -batch "matlab_detect_licks('trace.mat','ML Detection MATLAB Code/lickNets.mat','out.mat')"

% detectLicksFromRaw and its helpers live in the MATLAB code directory. Add it to the path so
% this script can be invoked from the repository root.
addpath(fullfile(pwd, 'ML Detection MATLAB Code'));

S = load(inMat);                       % expects S.rawData = [nSamples x 2] (time, capacitance)
lickTimes = detectLicksFromRaw(S.rawData, netFile);   % the ported-from cascade, unchanged

% Save as MATLAB v7 (readable by scipy.io.loadmat on the Python side).
save(outMat, 'lickTimes', '-v7');
fprintf('MATLAB_DETECT_DONE %d licks\n', numel(lickTimes));
end

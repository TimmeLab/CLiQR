% lickDetector
%
%   This script processes home-cage drinking monitor data to detect licks
%   and organize data for further processing. It is meant to work with the
%   capacitive lick system.
%
%   This script requires up to two additional excel data sheets. First, it
%   requires a sensor mapping file that relates sensors to animal names.
%   This file should consist of two columns, each with a header. The first
%   column is sipper labels. Each cell below the header should have
%   a name for a sipper or bottle that associates it with a given animal
%   (i.e., AEW4-1 or AEW4-1R). The second column is the sensor number on
%   the drinking system. Example sensor map file format:
%
%   Sipper Labels            Drinking Sensor
%   AEW4-1                   1
%   AEW4-2                   2
%   AEW4-3                   3
%
%   This would be the map for animal AEW4-1 using sensor 1, AEW4-2 using
%   sensor 2, and so forth.
%
%   The second the data sheet will be used if there are errors in the start
%   and stop times. The user will be able to override the start/stop times
%   recorded and use the capacitance traces to identify start and stop
%   times. This file will have four columns, each with headers: File
%   Number, Sensor, New Start Time, New End Time. The code will produce
%   plots for the user and data tips can be used identify new start and
%   stop times. Example time fix file format:
%
%   File Number    Sensor    New Start Time      New End Time
%   2              20        1756825855          1756833056
%   5              2         1757083015          1757090258
%
%   This would be time fixes for two recordings.
%
%   The script will produce an exclude variable to document recordings that
%   should be excluded because their number of licks and volume consumed
%   measures were not related similarly to other recordings. Each row of
%   excluded is a recording that should be excluded. The second column is
%   the file number (usually the date) and the first column is the index
%   of the sensor label (usually the animal). So, if excluded(1,:) = [3,2],
%   the second recording from sensor label 3 should be removed.
%
%   Version 1

%   Version History
%   1: Created in October 2025 to process data using a threshold procedure.


%% Identify the Data

% Ask the user to identify the files to process
[rawFiles,dataDir] = uigetfile('.h5','Select raw data files to process.','MultiSelect','on');
if isequal(rawFiles,0)
    disp('No raw files selected.')
    return
% If we have a return type of 'char', only one file was selected. Handle
% that here
elseif isequal(class(rawFiles), 'char')
    rawFiles = {rawFiles};
end
nFiles = length(rawFiles);

% Ask the user to identify the sensor mapping file
[sensorMapFileName,sensorMapDir] = uigetfile('.xlsx','Select the sensor map excel file.','MultiSelect','off');
if isequal(sensorMapFileName,0)
    disp('No sensor map file selected.')
    return
end
sensorMap = readcell(strcat(sensorMapDir,sensorMapFileName));
sipperLabels = sensorMap(2:end,1);
sensors = cell2mat(sensorMap(2:end,2));
nSensors = length(sensors);

% In the future add functionality for the user to pick different lick 
% detection algorithms here.

%% Load the raw data

% Load the data
data = cell([nFiles,nSensors,10]);
f = waitbar(0,'Loading Raw Data');

for iFile = 1:nFiles
    for iSensor = 1:nSensors
        for iBoard = 0:3
            try
                data{iFile,iSensor,1} = h5read(strcat(dataDir,rawFiles{iFile}),strcat("/board_FT232H",num2str(iBoard),"/sensor_",num2str(sensors(iSensor)),"/cap_data"));
                data{iFile,iSensor,2} = h5read(strcat(dataDir,rawFiles{iFile}),strcat("/board_FT232H",num2str(iBoard),"/sensor_",num2str(sensors(iSensor)),"/time_data"));
                data{iFile,iSensor,3} = h5read(strcat(dataDir,rawFiles{iFile}),strcat("/board_FT232H",num2str(iBoard),"/sensor_",num2str(sensors(iSensor)),"/start_time"));
                data{iFile,iSensor,4} = h5read(strcat(dataDir,rawFiles{iFile}),strcat("/board_FT232H",num2str(iBoard),"/sensor_",num2str(sensors(iSensor)),"/stop_time"));
                data{iFile,iSensor,5} = h5read(strcat(dataDir,rawFiles{iFile}),strcat("/board_FT232H",num2str(iBoard),"/sensor_",num2str(sensors(iSensor)),"/start_vol"));
                data{iFile,iSensor,6} = h5read(strcat(dataDir,rawFiles{iFile}),strcat("/board_FT232H",num2str(iBoard),"/sensor_",num2str(sensors(iSensor)),"/stop_vol"));
                data{iFile,iSensor,7} = h5read(strcat(dataDir,rawFiles{iFile}),strcat("/board_FT232H",num2str(iBoard),"/sensor_",num2str(sensors(iSensor)),"/weight"));
            end
        end
        waitbar((((iFile - 1)*nSensors) + iSensor)/(nFiles*nSensors),f,'Loading Raw Data')
    end
end
close(f)

%% Help the user fix sensor time errors

% Ask the user how long the recording should be in minutes
recLength = inputdlg('Input the desired recording length in minutes.');
if isempty(recLength)
    error('Recording length must be entered.')
end
recLength = str2double(cell2mat(recLength))*60; % Convert to seconds.

% Look for time log errors from short recordings
potentialTimeErrorFlag = 0;
for iFile = 1:nFiles
    for iSensor = 1:nSensors
        if (data{iFile,iSensor,4} - data{iFile,iSensor,3}) < recLength
            potentialTimeErrorFlag = 1;
        end
    end
end

% Inform the user of the error
if potentialTimeErrorFlag == 1
    f = msgbox('Some recordings were too short. Plots from recordings that are too short will be shown. Create a time fix file.');
    uiwait(f)

    % This loop will detect bad start/end times and plot to help the user
    % correct them. If all times are good, no plots will be produced.
    for iFile = 1:nFiles
        for iSensor = 1:nSensors
            if (data{iFile,iSensor,4} - data{iFile,iSensor,3}) < recLength
                figure
                h = plot(data{iFile,iSensor,2},data{iFile,iSensor,1});
                title(strcat('File: ',num2str(iFile),', Sensor: ',num2str(iSensor)))
                h.DataTipTemplate.DataTipRows(1).Format = '%.15f'; % Provide more digits for time values
            end
        end
    end

    f = msgbox('Click OK once the time fix file has been created.');
    uiwait(f)

    % Have the user identify the time fix file and load it.
    [timeFixFileName,timeFixDir] = uigetfile('.xlsx','Select the sensor map excel file.','MultiSelect','off');
    if isequal(timeFixFileName,0)
        disp('No time fix file selected.')
        return
    end
    timeFix = readcell(strcat(timeFixDir,timeFixFileName));
    timeFix = cell2mat(timeFix(2:end,:));
    
    % Update the start and end times with the fixes from the user
    for iFix = 1:size(timeFix,1)
        data{timeFix(iFix,1),timeFix(iFix,2),3} = timeFix(iFix,3);
        data{timeFix(iFix,1),timeFix(iFix,2),4} = timeFix(iFix,4);
    end

end

%% Add time adjusted capacitance data

for iFile = 1:nFiles
    for iSensor = 1:nSensors
        data{iFile,iSensor,8} = double(data{iFile,iSensor,1}((data{iFile,iSensor,2} > data{iFile,iSensor,3}) & (data{iFile,iSensor,2} < data{iFile,iSensor,4})));
        data{iFile,iSensor,9} = double(data{iFile,iSensor,2}((data{iFile,iSensor,2} > data{iFile,iSensor,3}) & (data{iFile,iSensor,2} < data{iFile,iSensor,4})));
        data{iFile,iSensor,9} = data{iFile,iSensor,9} - data{iFile,iSensor,9}(1);
    end
end

%% Detect licks

% Set the maximum duration of the lick threshold crossing in seconds
maxT = 1/6;

% Identify and characterize peaks at different thresholds
peakInfo = cell([nFiles,nSensors,2]);
f = waitbar(0,'Detecting Licks');
for iFile = 1:nFiles
    for iSensor = 1:nSensors
        cap = double(data{iFile,iSensor,8});
        T = data{iFile,iSensor,9};
        
        % Because the capacitance values are discretized, we can focus only
        % on the thresholds between the discrete values.
        uniqVals = unique(cap);
        if length(uniqVals) > 3
            thresholds = mean([uniqVals(1:(end - 1))';uniqVals(2:end)']);
            nPeaks = NaN([1,length(thresholds)]);
            peakBins = cell([length(thresholds),1]);
            for iThresh = 3:length(thresholds)

                % Identify peaks below threshold
                peakTrans = diff(cap < thresholds(iThresh)); % 1 marks the time bin just before a peak starts (the last bin that isn't in the peak), -1 marks the time bin just before a peak ends (the last bin in the peak)
                peakStarts = find(peakTrans == 1) + 1;
                peakEnds = find(peakTrans == -1);
                if cap(1) < thresholds(iThresh)
                    peakStarts = [1;peakStarts];
                end
                if cap(end) < thresholds(iThresh)
                    peakEnds = [peakEnds;length(cap)];
                end

                % Remove peaks that are below threshold for too long
                peakT = T(peakEnds) - T(peakStarts);
                goodPeaks = find(peakT < maxT);

                % Only look at peaks that are at least two thresholds
                % deep
                iPeak = 1;
                while iPeak <= length(goodPeaks)
                    if nnz(cap(peakStarts(goodPeaks(iPeak)):peakEnds(goodPeaks(iPeak))) < thresholds(iThresh - 2)) == 0
                        goodPeaks(iPeak) = [];
                    else
                        iPeak = iPeak + 1;
                    end
                end
                
                nPeaks(iThresh) = length(goodPeaks);
                
                % Record the peak time bins
                tempPeakTimes = NaN([length(goodPeaks),1]);
                for iPeak = 1:length(goodPeaks)
                    peakProfile = cap(peakStarts(goodPeaks(iPeak)):peakEnds(goodPeaks(iPeak)));
                    tempPeakTimes(iPeak) = find(peakProfile == min(peakProfile),1,'first') + peakStarts(goodPeaks(iPeak)) - 1;
                end
                peakBins{iThresh} = tempPeakTimes;


            end
            peakInfo{iFile,iSensor,1} = [thresholds;nPeaks];
            peakInfo{iFile,iSensor,2} = peakBins;
        end
        waitbar((((iFile - 1)*nSensors) + iSensor)/(nFiles*nSensors),f,'Detecting Licks')
    end
end
close(f)

% Select the threshold with the largest number of peaks and record the peak
% times
for iFile = 1:nFiles
    for iSensor = 1:nSensors
        iThresh = find(peakInfo{iFile,iSensor,1}(2,:) == max(peakInfo{iFile,iSensor,1}(2,:)),1,'first');
        data{iFile,iSensor,10} = data{iFile,iSensor,9}(peakInfo{iFile,iSensor,2}{iThresh});
    end
end

%% Detect outliers and potential bad recordings for exclusion

% Calculate the volume consumed, the dose, and the animal weight
volConsumed = NaN([nFiles,nSensors]);
animWeight = NaN([nFiles,nSensors]);
for iFile = 1:nFiles
    for iSensor = 1:nSensors
        volConsumed(iFile,iSensor) = data{iFile,iSensor,5} - data{iFile,iSensor,6};
        animWeight(iFile,iSensor) = data{iFile,iSensor,7};
    end
end

% Get the lick times and total lick counts
lickTimes = cell([nFiles,nSensors]);
nLicks = NaN([nFiles,nSensors]);
for iFile = 1:nFiles
    for iSensor = 1:nSensors
        licks = data{iFile,iSensor,10};
        licks(licks > recLength) = [];
        lickTimes{iFile,iSensor} = licks;
        nLicks(iFile,iSensor) = length(licks);
    end
end

% Fit a regular linear model
x = nLicks(:);
y = volConsumed(:);
mdl = fitlm(x,y);

% Fit a robust linear model
mdlr = fitlm(x,y,'RobustOpts','on');

% Plot residuals
% tiledlayout(1,2)
% nexttile
% plotResiduals(mdl,'probability')
% title('Linear Fit')
% nexttile
% plotResiduals(mdlr,'probability')
% title('Robust Fit')

% Detect outliers
outlier = find(isoutlier(mdlr.Residuals.Raw));

% Show the outliers to the user
if ~isempty(outlier)
    figure
    hold on
    scatter(x,y)
    scatter(x(outlier),y(outlier),'x','MarkerEdgeColor','r')
    title('Outliers detected.')
    xlabel('Number of Licks')
    ylabel('Volume Consumed (mL)')

    % Reorganize information about the outliers
    exclude = [ceil(outlier/nFiles),rem(outlier,nFiles)];
    exclude(exclude(:,2) == 0,2) = nFiles;

    % Plot the capacitance traces for excluded channels
    for iExclude = 1:size(exclude,1)
        figure
        plot(data{exclude(iExclude,2),exclude(iExclude,1),9},data{exclude(iExclude,2),exclude(iExclude,1),8})
    end
else
    exclude = [];
end





%% Package the lick data for the user

% Transpose the data files to match the sipper labels
lickTimes = lickTimes';
animWeight = animWeight';
volConsumed = volConsumed';
nLicks = nLicks';

% Ask the user where to save the file
[file,location] = uiputfile('','Save File');

% Save the file
if ~isempty(file)
    [filepath,file,ext] = fileparts(strcat(location,file));
    save(strcat(location,file,'.mat'),'lickTimes','animWeight','volConsumed','nLicks','exclude','rawFiles','sipperLabels','sensors')
end




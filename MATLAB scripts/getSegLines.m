function [segRPE, segCS] = getSegLines(averaged_dopu,savepath)
    [depth, numAscans, numBscans] = size(averaged_dopu); 
    
    % Parameters
    noiseSlices = 15;           % Number of noisy slices at beginning/end
    smoothingKernelSize = 15;   % Smoothing parameter for boundaries
    minGap = 30;                % Minimum distance between boundaries
    outlierThreshold = 3.0;     % MAD multiplier for outlier detection
    edgeWidth = round(numAscans * 0.15);  % Edge region width (15% of width)
    
    segRPE = zeros(numAscans,numBscans); % RPE
    segCS = zeros(numAscans,numBscans); % choroid-sclera junction
    
    for i = 1+noiseSlices:numBscans-noiseSlices
        
        if mod(i, 50) == 0
            fprintf('Processing slice %d of %d\n',i,numBscans);
        end
    
        currentSlice = averaged_dopu(:,:,i);
        
        % Simple thresholding
        thresholdValue = max(currentSlice(:)) - 0.001;
        binarySlice = currentSlice < thresholdValue;
        
        % Clean up binary image
        binarySlice = bwareaopen(binarySlice, 15);  % Remove small objects
        binarySlice = imfill(binarySlice, 'holes'); % Fill holes
    
        binarySlice = medfilt2(binarySlice, [15 15]);  % Adjust kernel size as needed
    
        for j = 1:numAscans
            idx = find(binarySlice(:,j));  % get all nonzero indices along the y-axis
            
            if ~isempty(idx)
                segRPE(j,i) = idx(1);      % first nonzero
                segCS(j,i) = idx(end);    % last nonzero
                
                span = idx(end) - idx(1); % detect abnormally large spans (likely outliers)
                if span > depth * 0.3 
                    segRPE(j,i) = NaN;
                    segCS(j,i) = NaN;
                end
            else
                segRPE(j,i) = NaN;         % NaN if no segmentation found
                segCS(j,i) = NaN;
            end
        end
    
         % Second-pass outlier detection using local context
        % This helps identify points that are inconsistent with their neighbors
        upperRaw = segRPE(:,i);
        lowerRaw = segCS(:,i);
        
        for col = 2:numAscans-1
            % For upper boundary - check if point deviates from local trend
            if ~isnan(upperRaw(col))
                localCols = max(1,col-5):min(numAscans,col+5);
                localCols = localCols(localCols ~= col); % Exclude current column
                localVals = upperRaw(localCols);
                validLocals = ~isnan(localVals);
                
                if sum(validLocals) >= 3
                    localMedian = median(localVals(validLocals));
                    localMAD = median(abs(localVals(validLocals) - localMedian));
                    localMAD = max(localMAD, 3);
                    
                    if abs(upperRaw(col) - localMedian) > 2.5 * localMAD
                        segRPE(col,i) = NaN;
                    end
                end
            end
            
            % For lower boundary - similar local context check
            if ~isnan(lowerRaw(col))
                localCols = max(1,col-5):min(numAscans,col+5);
                localCols = localCols(localCols ~= col); % Exclude current column
                localVals = lowerRaw(localCols);
                validLocals = ~isnan(localVals);
                
                if sum(validLocals) >= 3
                    localMedian = median(localVals(validLocals));
                    localMAD = median(abs(localVals(validLocals) - localMedian));
                    localMAD = max(localMAD, 3);
                    
                    if abs(lowerRaw(col) - localMedian) > 2.5 * localMAD
                        segCS(col,i) = NaN;
                    end
                end
            end
        end
    
    %     segRPE(:,i) = medfilt1(segRPE(:,i), 11);   % kernel size 9 (or try 7, 11 for tuning)
    %     segCS(:,i) = medfilt1(segCS(:,i), 11);
    
        % Interpolate missing values with polynomial fitting for U-shape preservation
        % Upper boundary
        validPoints = ~isnan(segRPE(:,i));
        if sum(validPoints) > numAscans/3
            validX = find(validPoints);
            validY = segRPE(validPoints, i);
            
            % Use polynomial fitting for the interpolation
            p = polyfit(validX, validY, 3);  % 3rd degree polynomial for flexibility
            fittedUpper = polyval(p, 1:numAscans);
            
            % For missing values, use the fitted curve
            for col = 1:numAscans
                if isnan(segRPE(col, i))
                    segRPE(col,i) = fittedUpper(col);
                end
            end
        else
            % Not enough points for fitting, use a default curve
            segRPE(:,i) = linspace(depth/3, depth/3, numAscans);
        end
        
        % Lower boundary - similar approach with polynomial fitting
        validPoints = ~isnan(segCS(:,i));
        if sum(validPoints) > numAscans/3
            validX = find(validPoints);
            validY = segCS(validPoints,i);
            
            % Use polynomial fitting for the interpolation
            p = polyfit(validX, validY, 3);  % 3rd degree polynomial
            fittedLower = polyval(p, 1:numAscans);
            
            % For missing values, use the fitted curve
            for col = 1:numAscans
                if isnan(segCS(col,i))
                    segCS(col,i) = fittedLower(col);
                end
            end
        else
            % Not enough points for fitting, use a default curve
            segCS(:,i) = linspace(2*depth/3, 2*depth/3, numAscans);
        end
        
        % Smooth the boundaries
        segRPE(:,i) = smooth(segRPE(:,i), smoothingKernelSize);
        segCS(:,i) = smooth(segCS(:,i), smoothingKernelSize);
        
        % Ensure minimum gap between boundaries
        for col = 1:numAscans
            gap = segCS(col,i) - segRPE(col,i);
            if gap < minGap
                midPoint = (segRPE(col,i) + segCS(col,i)) / 2;
                segRPE(col,i) = midPoint - minGap/2;
                segCS(col,i) = midPoint + minGap/2;
            end
        end
    
        % Inter-slice consistency - use information from previous slice if available
        if i > 1+noiseSlices
            % Blend with previous slice (80% current, 20% previous)
            segRPE(:,i) = 0.8 * segRPE(:,i) + 0.2 * segRPE(:,i-1);
            segCS(:,i) = 0.8 * segCS(:,i) + 0.2 * segCS(:,i-1);
        end
    end
    
    %% Convert segmentation lines to 3D for visualization
%     segRPEvol = zeros(depth, numAscans, numBscans);
%     segCSvol = zeros(depth, numAscans, numBscans);
%     
%     segRPE(segRPE<1) = 1; % fix interpolation outside of [1 depth]
%     segCS(segCS>depth) = depth;
%     
%     for i = 1:numBscans
%         for j = 1:numAscans
%             rowRPE = round(segRPE(j,i));  % Get the row index of the seg line
%             rowCS = round(segCS(j,i));
%             if rowRPE >= 1 && rowRPE <= depth
%                 segRPEvol(rowRPE,j,i) = 1;  % Set that point to 1 in the 3D mask
%             end
%             if rowCS >= 1 && rowCS <= depth
%                 segCSvol(rowCS,j,i) = 1;  % Set that point to 1 in the 3D mask
%             end
%         end
%     end
%     
%     for i=50:numBscans-49
%         overlay = imfuse(averaged_dopu(:,25:end-25,i),segRPEvol(:,25:end-25,i));
%         overlay2 = imfuse(overlay,segCSvol(:,25:end-25,i));
%         imshow(overlay2)
%         title(i)
%         pause(0.01)
%     end
% %     
    %% Export segmentation lines to ITK-snap for manual corrections
%     label_vol = zeros(depth, numAscans, numBscans);
%     
%     for i = 1:numBscans 
%         for j = 1:numAscans
%             rowRPE = round(segRPE(j,i));  % Get the row index of the seg line
%             rowCS = round(segCS(j,i));
%             if rowRPE >= 1 && rowRPE <= depth
%                 label_vol(rowRPE,j,i) = 1;  % Set that point to 1 in the 3D mask
%             end
%             if rowCS >= 1 && rowCS <= depth
%                 label_vol(rowCS,j,i) = 2;  % Set that point to 1 in the 3D mask
%             end
%         end
%     end
%     
%     for i=1:600
%         imadjusted_vol(:,:,i) = imadjust(mat2gray(averaged_oct(:,:,i)));
%     end
%     
%     niftiwrite(label_vol,fullfile(savepath,'seg_export'),'Compressed',true)
%     niftiwrite(imadjusted_vol,fullfile(savepath,'averaged_oct'),'Compressed',true)
%     niftiwrite(averaged_dopu,fullfile(savepath,'averaged_dopu'),'Compressed',true)
    
end













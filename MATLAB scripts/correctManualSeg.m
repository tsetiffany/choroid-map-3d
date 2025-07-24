function [top_line,bottom_line] = correctManualSeg(filepath)
    
    seg_corr = niftiread(filepath);
     
    [x, y, z] = size(seg_corr);
    top_line = zeros(y,z);
    bottom_line = zeros(y,z);
    
    % Extract top and bottom pixel indices for each A-scan
    for i = 1:z 
        for j = 1:y 
            seg_indices = find(seg_corr(:,j,i)); % get segmentation line(s)
            if ~isempty(seg_indices)
                top_line(j,i) = seg_indices(1);
                bottom_line(j,i) = seg_indices(end);
            end
        end
    end
    
    top_line = fillmissing(top_line, 'nearest');
    bottom_line = fillmissing(bottom_line, 'nearest');
end
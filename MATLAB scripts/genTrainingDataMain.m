%% Parameters

output_filepath = 'I:\2. CVI\15May25_MJ_OD\Seg\';
train_filepath = 'H:\CVI Training Data Set';

%% Optimally Oriented Flux (OOF)

tic
% radii = 2:2:28;
% opts = struct();
% opts.sigma = 2.5;              % pre-smoothing
% opts.spacing = [2 2 2];      % voxel size
% opts.useabsolute = 1;        % sort eigenvalues by magnitude
% opts.responsetype = 5;       % vesselness
% opts.normalizationtype = 1;  % normalization
% 
% [depth, numAscans, numBscans] = size(averaged_dopu); 

% for i=1:numBscans
%     adj_norm_dopu(:,:,i) = imadjust(mat2gray(averaged_dopu(:,51:end-50,i)));
% end

% dopu_OOF2 = oof3response(averaged_dopu, radii, opts);

radii = 2:2:32;
opts = struct();
opts.sigma = 2.5;              % pre-smoothing
opts.spacing = [2 2 2];      % voxel size
opts.useabsolute = 1;        % sort eigenvalues by magnitude
opts.responsetype = 5;       % vesselness
opts.normalizationtype = 1;  % normalization

dopu_OOF_flattened = oof3response(dopu_flattened, radii, opts);
toc

% holdvar = dopu_OOF2;
dopu_OOF2 = dopu_OOF_flattened;

figure; imshow(imadjust(mat2gray(dopu_OOF2(:,51:end-50,200))));

figure; imshow(imadjust(mat2gray(squeeze(mean(dopu_OOF2(:,51:end-50,51:end-50))))))

%% Generate binarized volume based on OOF

% for i=1:size(dopu_OOF2,3)
%     dopu_OOF2_norm = imadjust(mat2gray(dopu_OOF2(:,51:end-50,i)));
%     bw(:,:,i) = phansalkar(dopu_OOF2_norm, [16 16], 0.1);
%     bw(:,:,i) = dopu_OOF2_norm >= 0.5;
% end

for i=1:size(dopu_OOF2,3)
    dopu_OOF2_norm = imadjust(mat2gray(dopu_OOF2(:,51:end-50,i)));
    bw_small = phansalkar(dopu_OOF2_norm, [6 6], 0.0005);
    bw_large = phansalkar(dopu_OOF2_norm, [20 20], 0.1);
    bw(:,:,i) = bw_small & bw_large;
end


% for i=200%1:size(dopu_OOF2,3)
%     bw_test(:,:,i) = bwareaopen(bw(:,:,i),30);
% end
% 
% se = strel('disk',1);
% for i=1:size(dopu_OOF2,3)
%     bw_filled(:,:,i) = imdilate(bw_test(:,:,i),se);
% end
% 
% large_only = bw_large | bw_small;
% large_filled = imfill(large_only,'holes');
% bw_final = (bw(:,:,200) | large_filled) & large_only;

figure; imshowpair(imadjust(mat2gray(oct_flattened(:,51:end-50,200))),bw(:,:,200))
figure; imshowpair(imadjust(mat2gray(averaged_oct(:,51:end-50,200))),bw_filled(:,:,200))
figure; imshow(imadjust(mat2gray(oct_flattened(:,51:end-50,200))))

for i=51:size(bw,3)-50
    imshow(bw(:,:,i))
    pause(0.01)
end

figure; imshow(imadjust(mat2gray(squeeze(mean(bw(:,:,51:end-50))))))
% save(fullfile(output_filepath,'binarized_choroid'), 'bw', '-v7.3');

%% Apply DOPU segmentation lines to binarized volume and set out of bounds to zero
RPE_offset = 12;
CS_offset = -15;

segRPE_cropped = segRPE(51:end-50,:);
segCS_cropped = segCS(51:end-50,:);

for i=1:numBscans
    for j=1:numAscans-100
        upper = round(segRPE_cropped(j,i)) + RPE_offset;
        lower = round(segCS_cropped(j,i)) + CS_offset;

        % Basic bounds check
        if isnan(upper) || isnan(lower) || lower <= upper
            continue;
        end

        % Mask above upper
        bw(1:upper,j,i) = 0; 
        % Mask below lower
        bw(lower:end,j,i) = 0;
    end
end

%% Convert segmentation lines to 3D

segRPEvol = zeros(depth, numAscans-100, numBscans);
segCSvol = zeros(depth, numAscans-100, numBscans);

segRPE_cropped(segRPE_cropped<1) = 1; % fix interpolation outside of [1 depth]
segCS_cropped(segCS_cropped>depth) = depth;

for i = 1:numBscans
    for j = 1:numAscans-100
        rowRPE = round(segRPE_cropped(j,i)) + RPE_offset;  % Get the row index of the seg line
        rowCS = round(segCS_cropped(j,i)) + CS_offset;
        if rowRPE >= 1 && rowRPE <= depth
            segRPEvol(rowRPE,j,i) = 1;  % Set that point to 1 in the 3D mask
        end
        if rowCS >= 1 && rowCS <= depth
            segCSvol(rowCS,j,i) = 1;  % Set that point to 1 in the 3D mask
        end
    end
end
%% Display

% Check segmentation overlay with original DOPU
for i=50:numBscans-49
    overlay = imfuse(averaged_dopu(:,51:end-50,i),segRPEvol(:,:,i));
    overlay2 = imfuse(overlay,segCSvol(:,:,i));
    imshow(overlay2)
    title(i)
    pause(0.01)
end

% Check binarized choroid overlay with OCT
for i=50:numBscans-49
    imshowpair(imadjust(mat2gray(oct_flattened(:,51:end-50,i))),bw(:,:,i))
end

for i=50:numBscans-49
    imshowpair(imadjust(mat2gray(dopu_flattened(:,51:end-50,i))),bw(:,:,i))
end

% Enface projection of binarized volume after segmentation lines
figure; imshow(imadjust(mat2gray(squeeze(mean(bw(:,:,51:end-50))))))

%% Adaptive histogram equalization 

enf = mat2gray(squeeze(mean(bw(:,:,51:end-50),'omitnan')));
enf_histeq = adapthisteq(enf,'NumTiles',[50 50],'ClipLimit',0.05,'Distribution','exponential');

figure;imshow(imadjust(flipud(imrotate(enf_histeq,90)))); colormap gray
imwrite(imadjust(flipud(imrotate(enf_histeq,90))),fullfile(output_filepath,'choroid_enface.tif'));

for i=51:numBscans-50
    imwrite(uint8(255* imadjust(mat2gray(bw(:,:,i)))),fullfile(output_filepath,'\binarized_choroid.tif'),'WriteMode','append')
end

avg_oct_cropped = averaged_oct(:,51:end-50,:);
overlayOCT = avg_oct_cropped.*bw;

for i=1:600
    imshow(imadjust(mat2gray(overlayOCT(:,:,i))))
end

%% DNN vessel segmentation

enface_resize = imresize(enf_histeq,[1024,1024]);

DNN_model = "C:\Users\tiffa\Documents\1. Projects\CVI\choroid-map-3d\MATLAB scripts";
avg_dnn_all = DNN_Vessel_Segmentation_Updated(enface_resize, DNN_model);

avg_dnn_all(avg_dnn_all < 0.5) = 0;
avg_dnn_all(avg_dnn_all >= 0.5) = 1;

figure;imshow(imadjust(flipud(imrotate(avg_dnn_all,90))));

%% Save training dataset

octDir = fullfile(train_filepath, 'oct');
maskDir = fullfile(train_filepath, 'mask');

% Create directories if they don't exist
if ~exist(octDir, 'dir'), mkdir(octDir); end
if ~exist(maskDir, 'dir'), mkdir(maskDir); end

j = 0;
% Loop through each frame
for i=51:numBscans-50
    j = j+1;
    % Extract frame
    image = fixed(:,51:end-50,i);   % raw OCT
    mask  = bw(:,:,i);  % binarized mask

    % Save OCT frame
    octFilename = fullfile(octDir, sprintf('oct_%03d.mat', j));
    save(octFilename, 'image');

    % Save mask frame
    maskFilename = fullfile(maskDir, sprintf('mask_%03d.mat', j));
    save(maskFilename, 'mask');
end

disp('Saving complete!');



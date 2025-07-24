%% Parameters

output_filepath = 'I:\1. Reg_Manuscript\MJ_20250618_FOV_reg\LFOV\Seg';
seg_path = "I:\1. Reg_Manuscript\MJ_20250618_FOV_reg\LFOV\Seg\MJ_0618_seg_export_corr.nii.gz";
% train_filepath = 'H:\CVI Training Data Set';

[depth, numAscans, numBscans] = size(averaged_dopu); 
[segRPE, segCS] = getSegLines(averaged_dopu, output_filepath);

for i=1:numBscans
    imshow(imadjust(mat2gray(averaged_dopu(:,:,i))))
end

figure; imshow(imadjust(mat2gray(averaged_dopu(:,:,200))))

%% Flatten volumes based on DOPU RPE line
% AFTER MANUAL CORRECTION
[segRPE_corrected, segCS_corrected] = correctManualSeg(seg_path);
dopu_flattened = averaged_dopu;
oct_flattened = averaged_oct;

target_row = 350; % change this depending on volume
segRPE_flattened = segRPE;
segCS_flattened = segCS_corrected;

for i=1:numBscans
    for j = 1:numAscans
        shift = target_row - segRPE(j,i); % use original RPE; artifacts with manually corrected one
        dopu_flattened(:,j,i) = circshift(averaged_dopu(:,j,i), round(shift));
        oct_flattened(:,j,i) = circshift(oct_flattened(:,j,i), round(shift));
        segRPE_flattened(j,i) = segRPE(j,i) + round(shift);
        segCS_flattened(j,i) = segCS_corrected(j,i) + round(shift); % flatten the corrected CS line
    end
end

for i=1:numBscans
    imshow(imadjust(mat2gray(oct_flattened(:,:,i))))
end
%% Optimally Oriented Flux (OOF)

tic
% Large vessel setting
radii = 5:5:40;
opts = struct();
opts.sigma = 2.5;              % pre-smoothing
opts.spacing = [2 2 2];      % voxel size
opts.useabsolute = 1;        % sort eigenvalues by magnitude
opts.responsetype = 5;       % vesselness
opts.normalizationtype = 1;  % normalization

dopu_OOF = oof3response(dopu_flattened, radii, opts);

figure; imshow(imadjust(mat2gray(dopu_OOF(:,51:end-50,200))));
figure; imshow(imadjust(mat2gray(squeeze(mean(dopu_OOF(:,51:end-50,51:end-50))))))

% Small vessel setting
radii = 1:3:13;
opts = struct();
opts.sigma = 1;              % pre-smoothing
opts.spacing = [2 2 2];      % voxel size
opts.useabsolute = 1;        % sort eigenvalues by magnitude
opts.responsetype = 5;       % vesselness
opts.normalizationtype = 1;  % normalization

dopu_OOF2 = oof3response(dopu_flattened, radii, opts);
toc

figure; imshow(imadjust(mat2gray(dopu_OOF2(:,51:end-50,200))));

figure; imshow(imadjust(mat2gray(squeeze(mean(dopu_OOF2(:,51:end-50,51:end-50))))))

%% Generate binarized volume based on OOF

for i=1:size(dopu_OOF,3)
    OCT_OOF_norm_large = imadjust(mat2gray(dopu_OOF(:,51:end-50,i)));
    OCT_OOF_norm_small = imadjust(mat2gray(dopu_OOF2(:,51:end-50,i)));

    % Phansalkar thresholding by vessel size
    bw_large(:,:,i) = phansalkar(OCT_OOF_norm_large, [20 20], 0.1);
    bw_small(:,:,i) = phansalkar(OCT_OOF_norm_small, [8 8], 0.0005);
    bw_phkr(:,:,i) = bw_large(:,:,i) | bw_small(:,:,i);

    % Simple thresholding on DOPU and multiply with Phansalkar mask
    dopu_thresh = (imadjust(mat2gray(dopu_flattened(:,51:end-50,i)))) >= 0.5;
    bw(:,:,i) = dopu_thresh.*bw_phkr(:,:,i);

    % Morphological filter
    bw(:,:,i) = bwareaopen(bw(:,:,i),30); % remove chunks less than 30 pixels
end

for i=51:size(bw,3)-50
    imshowpair(imadjust(mat2gray(oct_flattened(:,51:end-50,i))),bw(:,:,i))
end

%% Visualize lines and determine offset 
figure;
imshow(bw(:,:,200)); hold on;
plot(1:500, segRPE_flattened(51:end-50,200),'r-','LineWidth',1.5);
plot(1:500, segCS_flattened(51:end-50,200),'b-','LineWidth',1.5);

%% Apply DOPU segmentation lines to binarized volume and set out of bounds to zero
RPE_offset = 10;
CS_offset = -10;

segRPE_cropped = segRPE_flattened(51:end-50,:);
segCS_cropped = segCS_flattened(51:end-50,:);

for i=1:numBscans
    for j=1:numAscans-100
        upper = round(segRPE_cropped(j,i)) + RPE_offset;
        lower = round(segCS_cropped(j,i)) + CS_offset;

        % Mask above upper
        bw(1:target_row + RPE_offset,j,i) = 0; 
        % Mask below lower
        bw(lower:end,j,i) = 0;
    end
end

%% Convert segmentation lines to 3D and imfuse

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

% Check segmentation overlay with original DOPU
for i=50:numBscans-49
    overlay = imfuse(dopu_flattened(:,51:end-50,i),segRPEvol(:,:,i));
    overlay2 = imfuse(overlay,segCSvol(:,:,i));
    imshow(overlay2)
    title(i)
    pause(0.01)
end

%% Display

% Check binarized choroid overlay with OCT
for i=50:numBscans-49
    imshowpair(imadjust(mat2gray(oct_flattened(:,51:end-50,i))),bw(:,:,i))
end

for i=50:numBscans-49
    imshowpair(imadjust(mat2gray(dopu_flattened(:,51:end-50,i))),bw(:,:,i))
end

for i=target_row:500
    imshow(imadjust(mat2gray(squeeze(bw(i,:,51:end-50)))))
    pause()
end

% Enface projection of binarized volume after segmentation lines
figure; imshow(flipud(imrotate(imadjust(mat2gray(squeeze(mean(bw(:,:,51:end-50))))),90)))

%% Adaptive histogram equalization 

enf = mat2gray(squeeze(mean(bw(:,:,51:end-50),'omitnan')));
enf_histeq = adapthisteq(enf,'NumTiles',[20 20],'ClipLimit',0.1,'Distribution','exponential');

figure;imshow(imadjust(flipud(imrotate(enf_histeq,90)))); colormap gray
imwrite(imadjust(flipud(imrotate(enf_histeq,90))),fullfile(output_filepath,'choroid_enface.tif'));

%% Undo flattening and save 
bw_unflattened = bw;

for i=1:numBscans
    for j = 1:numAscans-100
        shift = -(target_row - segRPE(j+50,i));
        bw_unflattened(:,j,i) = circshift(bw(:,j,i), round(shift));
    end
end

figure; imshow(imadjust(mat2gray(averaged_oct(:,51:end-50,200))))
figure; imshowpair(imadjust(mat2gray(averaged_oct(:,51:end-50,200))),bw_unflattened(:,:,200))

for i=51:numBscans-50
    imshowpair(imadjust(mat2gray(averaged_oct(:,51:end-50,i))),bw_unflattened(:,:,i))
end

for i=51:numBscans-50
    imshowpair(imadjust(mat2gray(averaged_dopu(:,51:end-50,i))),bw_unflattened(:,:,i))
    imwrite(uint8(255* imadjust(mat2gray(bw_unflattened(:,:,i)))),fullfile(output_filepath,'binarized_choroid.tif'),'WriteMode','append')
end

save(fullfile(output_filepath,'binarized_choroid'), 'bw_unflattened', '-v7.3');

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
    if mod(j, 50) == 0
        fprintf('Saving slice %d of %d\n',j,numBscans-100);
    end
    % Extract frame
    image = imadjust(mat2gray(fixed(:,51:end-50,i)));   % raw OCT
    mask  = bw_unflattened(:,:,i);  % binarized mask

    % Save OCT frame
    octFilename = fullfile(octDir, sprintf('oct_%03d.mat', j));
    save(octFilename, 'image');

    % Save mask frame
    maskFilename = fullfile(maskDir, sprintf('mask_%03d.mat', j));
    save(maskFilename, 'mask');
end

disp('Saving complete');



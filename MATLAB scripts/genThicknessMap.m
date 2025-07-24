filepath = 'I:\1. Reg_Manuscript\MJ_20250618_FOV_reg\LFOV\Seg';

% Compute thickness at each A-scan (en face location)
thickness_map = segCS_corrected(51:end-50,51:end-50) - segRPE_corrected(51:end-50,51:end-50);

valid_vals = thickness_map(~isnan(thickness_map));
mean_val = mean(valid_vals);
std_val = std(valid_vals);

% Set conservative bounds (e.g., 4 stds away to allow for pathology)
lower_bound = max(mean_val - 4*std_val, 1);
upper_bound = mean_val + 4*std_val;

% Mask outliers
outlier_mask = thickness_map < lower_bound | thickness_map > upper_bound;
thickness_map(outlier_mask) = NaN;
temp_map = fillmissing(thickness_map,'nearest');

smoothed_thickness_map = medfilt2(temp_map, [5 5]);

fig1 = figure;
imagesc(flipud(imrotate(smoothed_thickness_map,90)), [min(temp_map(:)), max(temp_map(:))]);
axis image off;
colormap jet;
colorbar;
title('Retinal Thickness Map (pixels)');
saveas(fig1, fullfile(filepath,'thickness_map_colorbar.png'));  % Saves in current folder

%% hsv with jet colourmap

thickness_norm = (smoothed_thickness_map - min(temp_map(:))) / (max(temp_map(:)) - min(temp_map(:)));
thickness_norm(isnan(thickness_norm)) = 0;

jetmap = jet(256);

% Map normalized thickness to colormap indices (1 to 256)
idx = round(thickness_norm * 255) + 1;  % ensure indices in [1,256]
idx(idx < 1) = 1;
idx(idx > 256) = 256;

rgb_img = zeros([size(thickness_norm), 3]);

% Assign jet colors pixel-wise
for c = 1:3
    channel = jetmap(idx, c);
    rgb_img(:,:,c) = reshape(channel, size(thickness_norm));
end

% Process vessel intensity (value channel)
value = notchfilter(enf_histeq);
value = medfilt2(value, [2 2]);
value(isnan(value)) = 0;  % Handle NaNs

% Modulate the RGB colors by vessel intensity
for c = 1:3
    rgb_img(:,:,c) = rgb_img(:,:,c) .* value;
end

figure;
imshow(flipud(imrotate(rgb_img,90)));
title('Thickness colored by jet colormap modulated by vessel intensity');
frame = getframe(gca);  % Capture current axes content
imwrite(frame.cdata, fullfile(filepath,'rgb_modulated_image.png'));

%% alpha jet
thickness_norm = (smoothed_thickness_map - min(temp_map(:))) / (max(temp_map(:)) - min(temp_map(:)));
enf = mat2gray(squeeze(mean(bw(:,:,51:end-50),'omitnan')));
enf_histeq = adapthisteq(enf,'NumTiles',[20 20],'ClipLimit',0.05,'Distribution','exponential');
colormap_jet = jet(256);
thickness_rgb = ind2rgb(uint8(thickness_norm * 255), colormap_jet);
alpha_mask = ~isnan(thickness_map);    
alpha_mask = double(alpha_mask) * 0.4;

figure;

% Show vessel image as background
imshow(enf_histeq, 'InitialMagnification', 'fit');
hold on;

% Overlay thickness map with transparency
h = imshow(thickness_rgb);
set(h, 'AlphaData', alpha_mask);

title('Thickness Map Overlaid on Choroidal Vessel En Face');

%% hsv
hue = (smoothed_thickness_map - min(temp_map(:))) / (max(temp_map(:)) - min(temp_map(:)));

% Use vessel for intensity (value channel)
value = notchfilter(enf_histeq);
value = medfilt2(value,[2 2]);
% value = imresize(avg_dnn_all, [500 500]);

% Saturation fixed or based on confidence
saturation = ones(size(hue));

% Handle NaNs: make hue and value = 0
hue(isnan(smoothed_thickness_map)) = 0;
value(isnan(smoothed_thickness_map)) = 0;
saturation(isnan(smoothed_thickness_map)) = 0;

% Combine into HSV image
hsv_img = cat(3, hue, saturation, value);
rgb_img = hsv2rgb(hsv_img);

figure;
imshow(rgb_img);
title('Hue = Thickness, Intensity = Choroidal Vessel Signal');

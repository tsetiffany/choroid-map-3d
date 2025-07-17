function im_seg = DNN_Vessel_Segmentation_Updated(im, Network_Location)
tic

new_net = load(fullfile(Network_Location, '/matlab_model.mat'));

octa_net = new_net.net;

im2 = imresize(mat2gray(double(im)), [1024 1024]);
pred = predict(octa_net, im2);
im_seg = pred(:,:,2);

toc
end
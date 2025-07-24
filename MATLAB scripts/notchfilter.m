function [filt] = notchfilter(img)
    imgfft = fftshift(fft2(img));
    imgfft2 = imgfft;
    
    size(img,1);
    xg1=1:size(img,1);
    xg2=1:size(img,1);
    ag = 1;
    bg1 = size(img,1)/2;
    bg2=size(img,1)/2;
    
    cg1 = 1.4;
    cg2 = 1;
    fg = exp(-(xg1-bg1).^2./(2*cg1^2));
    fg1 = exp(-(xg2-bg2).^2./(2*cg2^2));
    fg = abs(fg-1);
    
    ffg = repmat(fg,[size(img,1) 1]);
    ffg1 = repmat(fg1',[1 size(img,1)]);
    
    ffg2 = ffg+ffg1;
    ffg2(ffg2>1) = 1;
    fffg = imgfft2.*ffg2';
    filt = abs(ifft2(fffg));

end
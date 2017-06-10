function testCount(d,testData,b,dict)

nFrames = size(testData,1);

results = zeros(nFrames,2); %[GTcount EstimatedCount];

for f = 1:nFrames
  
  disp(['Testing on frame ' num2str(f) '/' num2str(nFrames)]);
  
  orgIm = imread(fullfile(d.datapath,'test',[testData{f,1} '.' d.imExt]));
  orgImSz = size(orgIm);
  
  im = imresize(orgIm,d.sFactor);
  
  imf = encodeImage(im,d);
  sz = size(imf);
  imf = shiftdim(imf,2);
  imf = reshape(imf,size(imf,1),[]);
  
  Idx = uint16(vl_kdtreequery(dict.tree,dict.means,imf));
  
  densityEst = reshape(b(Idx)+b(end),sz(1),sz(2));
  densityEst = vl_imsmooth(densityEst,d.sigma/2);
  globalCount = sum(densityEst(:));
    
  if ~isempty(testData{f,2}) %get ground truth count if available
    dots = testData{f,2};
    dots = round(dots*d.sFactor);
    density = getDensity(im,dots,d.sigma,d.cropSize);
    disp(['Frame: ' num2str(f) '--> #Dots: ' num2str(size(dots,1))]);
    GT = sum(density(:));
  else %There is no GT or no objects in the image
    GT = 0;
  end
  
  % store results for this frame
  results(f,:) = [GT globalCount];
  disp(['Density GT: ' num2str(GT) ' - Estimated: ' num2str(globalCount)]);
  
  if d.saveDensity
    orgDensityEst = padarray(densityEst,[d.cropSize d.cropSize]);
    orgDensityEst = imresize(orgDensityEst,[orgImSz(:,1) orgImSz(:,2)]);
    normCte = sum(orgDensityEst(:))/sum(densityEst(:));
    orgDensityEst = orgDensityEst/normCte;
    ext = strfind(testData{f,1},'.');
    if isempty(ext)
      nameend = numel(testData{f,1});
    else
      nameend = ext-1;
    end
    save(fullfile(d.exppath,[testData{f,1}(1:nameend) '.mat']),'orgDensityEst');
  end
  
  if d.segment %density-based segmentation
    orgDensityEst = padarray(densityEst,[d.cropSize d.cropSize]);
    orgDensityEst = imresize(orgDensityEst,[orgImSz(:,1) orgImSz(:,2)]);
    normCte = sum(orgDensityEst(:))/sum(densityEst(:));
    orgDensityEst = orgDensityEst/normCte;
    
    seg = segmentDensity(orgDensityEst,orgIm,d);
    
    ext = strfind(testData{f,1},'.');
    if isempty(ext)
      nameend = numel(testData{f,1});
    else
      nameend = ext-1;
    end
    imwrite(seg,fullfile(d.exppath,[testData{f,1}(1:nameend) '.jpg']));
  end
  
  
end %end stack testing

save(fullfile(d.exppath,'Results.mat'),'results');

figure;
plot([0 max(results(:))],[0 max(results(:))],'-r','linewidth',2);
hold on,
plot(results(:,1),results(:,2),'ob','linewidth',3);
xlabel('GT count','fontsize',14);
ylabel('Estimated count','fontsize',14);
title(['Crystal count prediction. Dictionary size: ' num2str(d.dictSize)...
  ' - # Train images: ' num2str(d.subsetSize)]...
  ,'fontsize',14);

end
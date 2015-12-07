indir = 'Z:\documents\CMU\16-720\project\720project\data\BSR\groundTruth\test\';
outdir = 'Z:\documents\CMU\16-720\project\720project\data\BSR\extracted_groundTruth\';
files = dir([indir, '*.mat']);
mkdir(outdir);
for file = files'
  load([indir, file.name]);
  I = groundTruth{1}.Boundaries;
  for i=2:length(groundTruth)
    I = I + groundTruth{i}.Boundaries;
  end
  imwrite(1-I./length(groundTruth), [outdir, file.name, '.jpg']);
end
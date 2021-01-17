clear all;
close all;
clc;

train_files = dir(fullfile('Yale2_train\', '*.bmp'));

for i = 1:length(train_files)
    imgBuffer = imread(fullfile('Yale2_train\', train_files(i).name));
    imgBuffer = im2double(imgBuffer);
    Data(:,i) = reshape(imgBuffer, [], 1);
end

MeanValue = mean(Data, 2);

avg_face = reshape(MeanValue,[100,100]);
figure;
imshow(avg_face);
title('Average Face');

OriginalMean = repmat(MeanValue,1, size(Data,2));
DataAdjust = Data - OriginalMean;

C1 = DataAdjust' * DataAdjust;
[C1_EigenVectors, C1_EigenValues] = eig(C1);
V = DataAdjust * C1_EigenVectors;

EigenVectors = V ./ (ones(size(V, 1),1) * sqrt(sum(V .* V)));

FeatureVectors = fliplr(EigenVectors);
figure;
for i = 1:9
	subplot(3, 3, i)
	imagesc(reshape(FeatureVectors(:,i),[100,100]));
	title(['Eigenface ' num2str(i)]);
end
colormap('gray');

PatternVectors = FeatureVectors' * DataAdjust;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_files = dir(fullfile('Yale2_test\', '*.bmp'));
success = 0;

for i = 1:length(test_files)
    TestFace = imread(fullfile('Yale2_test\', test_files(i).name));
    imgBuffer = im2double(TestFace);
    TestData = reshape(imgBuffer, [], 1);

    TestDataPatternVector = FeatureVectors' * (TestData - MeanValue);

    result = inf;
    posn = 0;
    

    for j = 1:length(train_files)
        temp = norm(PatternVectors(:,j) - TestDataPatternVector);
        if (temp < result)
            result = temp;
            posn = j;
        end
    end
    
    if (isequal(test_files(i).name(1:9),train_files(posn).name(1:9)))
        success = success + 1;
    else
        figure;
        subplot(1, 2, 1)
        imagesc(TestFace);
        title('Test Face');
    
        subplot(1, 2, 2)
        imagesc(reshape(Data(:,posn),[100,100]));
        title('Failed to recognize');
        colormap('gray');
    end
end

fprintf('Trained by %d',length(train_files));
fprintf(' samples\nTested by %d',length(test_files));
fprintf(' samples\nout of which %d',success);
fprintf(' succeeded  %d',length(test_files)-success); 
fprintf(' failed\nOverall accuracy %f',success*100/length(test_files));
fprintf('%%\n');



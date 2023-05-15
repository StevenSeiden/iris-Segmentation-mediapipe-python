clear
close all

users = ["Steven","Long","Kyle","Ruxin","Zeyu"];

choices_abc = ["A","B","C"];

XTrain = zeros(10*length(users)*length(choices_abc),(4*3)*100);
YTrain = strings(10*length(users)*length(choices_abc),1);

XTest = zeros(10*length(users)*length(choices_abc),(4*3)*100);
YTest = strings(10*length(users)*length(choices_abc),1);

for u = 1:length(users)
    choices ={strcat(users(u),' A'),strcat(users(u),' B'),strcat(users(u),' C')};

    %Path to data, assuming folder for each user
    folderName = strcat('/Users/stevenseiden/Developer/iris-Segmentation-mediapipe-python/',users(u),'/');

    for c = 1:length(choices)
        subFolderName = strcat(folderName,convertCharsToStrings(choices{c}),'/');
        files = dir(strcat(subFolderName,'trial*'));
        for f = 1:length(files)
            filename = strcat(files(f).folder,'/',files(f).name);
            data = readtable(filename);

            %% extract some features rather than using the raw data
            dataInterpolated = zeros(1,(4*3)*100);
            for i = 1:4
                temp = interp1([1:size(data,1)],data{:,i},[1:size(data,1)/110:size(data,1)]);
                temp = temp(1:100);
                % do normalization
                temp = (temp-temp(1))/(temp(length(temp))-temp(1))*(1-0);
                dataInterpolated(1+(i-1)*100:i*100) = temp;
                % add features
%                 dataInterpolated(1+(4+i-1)*100:(4+i)*100) = movstd(temp,20);
                dataInterpolated(1+(4+i-1)*100:(4+i)*100) = movvar(temp,100);
                windowLength = 70;
                for j = 1:100-windowLength+1
                    dataInterpolated(j+(8+i-1)*100) = skewness(temp(j:j+windowLength-1));
                end
            end

            if f <= 10
                XTrain(30*(u-1)+10*(c-1)+f,:) = dataInterpolated;
                YTrain(30*(u-1)+10*(c-1)+f,1) = choices_abc(c);
            else
                XTest(30*(u-1)+10*(c-1)+f-10,:) = dataInterpolated;
                YTest(30*(u-1)+10*(c-1)+f-10,1) = choices_abc(c);
            end
        end
    end
end

%% Try different models
model = fitcecoc(XTrain,YTrain);

YPredict = predict(model,XTest);

accuracy = mean(YPredict == YTest)

cm = confusionchart(YTest,string(YPredict));
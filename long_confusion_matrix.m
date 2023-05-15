function long_confusion_matrix(groundtruth,prediction,classes)

%classes = unique(groundtruth);

confusion_matrix = zeros(length(classes),length(classes));

for i = 1:length(classes)
    index = find(groundtruth == classes(i));
    predicted_labels = prediction(index);
    for j = 1:length(classes)
        confusion_matrix(i,j) = sum(predicted_labels == classes(j))/length(index);
    end
end

figure;
image(confusion_matrix,'CDataMapping','scaled');
c = flipud(gray);
c = c(floor(size(c,1)*0.1):floor(size(c,1)*0.8),:);
c(1,:) = ones(1,3);
colormap(c);

xticks(1:length(classes));
yticks(1:length(classes));

xticklabels(upper(classes));
yticklabels(upper(classes));

set(gca,'TickLength',[0 0]);

xlabel('Predicted Class');
ylabel('True Class');

for i = 1:length(classes)
    for j = 1:length(classes)
        num = round(confusion_matrix(i,j),2);
        first_digit = floor(num/1);
        second_digit = floor((num-first_digit*1)/0.1);
        third_digit = floor((num-first_digit*1-second_digit*0.1)/0.01);
        num_string = strcat(num2str(first_digit),'.',num2str(second_digit),num2str(third_digit));
        if confusion_matrix(i,j) > 0.5
            text(j,i,num_string,'HorizontalAlignment','center','color','white','fontsize',16,'FontWeight','bold');
        elseif confusion_matrix(i,j) >= 0
            text(j,i,num_string,'HorizontalAlignment','center','color','black','fontsize',16,'FontWeight','bold');
        end
    end
end

set(gca,'FontSize', 20);%,'Linewidth',2);
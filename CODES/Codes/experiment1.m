clear all; close; clc;




load  fisheriris;


%VÝSUALÝZÝNG Fisherisis Dataset
Future1 = meas(:,1); %   1. sütunu aldým
Future2=meas(:,2); %2.sütunu aldým.
result = gscatter(Future1,Future2,species,'krb','ov^',[],'off');
result(1).LineWidth = 2;
result(2).LineWidth = 2;
result(3).LineWidth = 2;
legend('Setosa','Versicolor','Virginica','Location','best');title('Fisheris Dataset');
hold on



%Spilitting Fishresis Dataset

X = meas(:,:);
Y = species;


[m,n] = size(X);
P = 0.80;
idx = transpose(randperm(m));
Training_X = X(idx(1:round(P*m)),:);
Training_Y= Y(idx(1:round(P*m)),:);
Testing_X = X(idx(round(P*m)+1:end),:);
Testing_Y = Y(idx(round(P*m)+1:end),:);



%%Using  SVM algorithm  for multiclass classification for Fisheriris

y=numel(unique(Training_Y));

temp = templateSVM('KernelFunction', 'linear');
model = fitcecoc(Training_X, Training_Y, 'Learner', temp); %Modelim train setle eðitiliyor
y_pred = predict(model, Testing_X); %Daha sonra modele testimi veriyorum ve prediction yapýyorum 


result_of_matrice= confusionmat(Testing_Y, y_pred); %Buradan skorlarýmý çekeceðim 
 
[Metric_Table] = metrics(result_of_matrice,y);
disp(Metric_Table)

%% 

%STANDARTÝZATÝON OF FÝSHERÝS DATASET

load fisheriris.mat
X = meas(:,:);
Y = species;
Mean_value=mean((mean(X)));
Standart_value=std((std(X)));
[w h]=size(X);
X_new=zeros(w,h);

for i=1:4
    X_new(:,i) = ((X(:,i)-Mean_value))/(Standart_value);
end

[m,n] = size(X);
Per = 0.80;
idx = transpose(randperm(m));
Training_X2= X_new(idx(1:round(Per*m)),:);
Training_Y2= Y(idx(1:round(Per*m)),:);
Testing_X2 = X(idx(round(Per*m)+1:end),:);
Testing_Y2= Y(idx(round(Per*m)+1:end),:);



model = fitcecoc(Training_X2, Training_Y2, 'Learner', temp);
y_pred2 = predict(model, Testing_X2);

result_of_matrice= confusionmat(Testing_Y2, y_pred2); %Buradan skorlarýmý çekeceðim 
 
[Metric_Table] = metrics(result_of_matrice,y);
disp('Metrics: ')
disp(Metric_Table)





%%

%DATA VÝZUALÝTÝON FOR IONOSPHERE DATASET

clear all; close; clc;
load ionosphere.mat

Future1 = X(:,3); %   1. sütunu aldým
Future2=X(:,4); %2.sütunu aldým.
result = gscatter(Future1,Future2,Y,'krb','ov^',[],'off');
result(1).LineWidth = 2;
result(2).LineWidth = 2;
legend('Good','Bad','Location','best');title('Ionesphere Dataset');
hold on


%Spilitting Ionosphere Dataset

%Part_X=X(:,end-20:end); %Bazý columnlarý çýkardým
[m n]=size(X);
P = 0.80;
idx = transpose(randperm(m));
Training_X =X(idx(1:round(P*m)),:);
Training_Y= Y(idx(1:round(P*m)),:);
Testing_X =X(idx(round(P*m)+1:end),:);
Testing_Y = Y(idx(round(P*m)+1:end),:);


%%Using  SVM algorithm  for multiclass classification for Fisheriris

y=numel(unique(Training_Y));

temp = templateSVM('KernelFunction', 'linear');
model = fitcecoc(Training_X, Training_Y, 'Learner', temp);
y_pred = predict(model, Testing_X);

result_of_matrice= confusionmat(Testing_Y, y_pred); %Buradan skorlarýmý çekeceðim 
 
[Metric_Table] = metrics(result_of_matrice,y);
disp('Metrics: ')
disp(Metric_Table)

%%
clear all; close; clc;
load ionosphere.mat

%STANDARDÝZATÝON OF IONOSPHERE DATASET
mean_value=mean(X);
stand_value=std(X);

[m n ]=size(X);
X_new=zeros(m,n);

for i=1:1:n
    
    X_new(:,i) = ((X(:,i)-mean_value))/(stand_value); %Burda il sutundaký tum elemanlardan sýrayla mean cýkarýp std bölerek 
end


%SVM For STANDARDÝZATE IONOSPHERE DATASET 


[m,n] = size(X_new);
Per = 0.80;
idx = transpose(randperm(m));
Training_X2= X_new(idx(1:round(Per*m)),:);
Training_Y2= Y(idx(1:round(Per*m)),:);
Testing_X2 = X_new(idx(round(Per*m)+1:end),:);
Testing_Y2= Y(idx(round(Per*m)+1:end),:);
    


%classes = unique(Training_Y2);
y=numel(unique(Training_Y2));

temp = templateSVM('KernelFunction', 'linear');
model = fitcecoc(Training_X2, Training_Y2, 'Learner', temp);
y_pred = predict(model, Testing_X2);

result_of_matrice= confusionmat(Testing_Y2, y_pred); %Buradan skorlarýmý çekeceðim 
 
[Metric_Table] = metrics(result_of_matrice,y);
disp('Metrics: ')
disp(Metric_Table)






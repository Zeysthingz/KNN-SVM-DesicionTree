function [Metric_Table] = metrics(c,y)


for i=1:y
   True_Positive(i) = c(i,i);
   False_Positive(i) = sum(c(:,i))-c(i,i);
   False_Negative(i) = sum(c(i,:))-c(i,i);
   True_Negative(i) = sum(sum(c))-True_Positive(i)-False_Positive(i)-False_Negative(i);       
end

 for i=1:y
 Accuracy(i) = ((True_Positive(i)+True_Negative(i))/(True_Positive(i)+False_Positive(i)+False_Negative(i)+True_Negative(i)))*100;
 Recall(i)=(c(i,i)/sum(c(i,:)))*100;
 Precision(i) = (c(i,i)/sum(c(:,i)))*100;
 end
 
 F1_score = (2*Precision(:).*Recall(:)./(Precision(:)+Recall(:)))';

for i = 1:y
   TPR(i) = (True_Positive(i)/(True_Positive(i)+False_Negative(i)))*100;
   FPR(i) = (False_Positive(i)/(False_Positive(i)+True_Negative(i)))*100;
end


Avg_Accuracy =mean(Accuracy);
Avg_Precision = mean(Precision);
Avg_Recall = mean(Recall);
Avg_F1 = mean(F1_score);
Avg_TPR = mean(TPR);
Avg_FPR = mean(FPR);

if y == 3
    
  T =table(Accuracy',Precision',Recall',F1_score',TPR',FPR','VariableNames',{'Accuracy','Precision','Recall','F1','TPR','FPR'},'RowNames',{'setosa';'versicolor';'virginica'});
  T1 =table(Avg_Accuracy,Avg_Precision,Avg_Recall,Avg_F1,Avg_TPR,Avg_FPR,'VariableNames',{'Accuracy','Precision','Recall','F1','TPR','FPR'},'RowNames',{'Average'});
  Metric_Table = [T;T1];
  

end

if y==2
  T =table(Accuracy',Precision',Recall',F1_score',TPR',FPR','VariableNames',{'Accuracy','Precision','Recall','F1','TPR','FPR'},'RowNames',{'Good';'Bad'});
  T1 =table(Avg_Accuracy,Avg_Precision,Avg_Recall,Avg_F1,Avg_TPR,Avg_FPR,'VariableNames',{'Accuracy','Precision','Recall','F1','TPR','FPR'},'RowNames',{'Average'});
  Metric_Table = [T;T1];
end
    


end
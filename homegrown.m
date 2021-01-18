training_to_test_ratio= .1:.1:.5;     %0.1/0.9 0.2/0.8 0.3/0.7 ... training to test

coding = {'onevsone', 'onevsall'};

disp(' ')
disp('---------------------------------------------------------------------------------------------------------- ')
disp('-----------------------------------------------------------------------------')
disp('  IRIS data')
myTable=readtable('Iris_data.txt');
X=table2array( myTable(:,1:4));
Y=table2array(myTable(:,5));
disp('Homegrown (Deliverable 2)')
F_LinearNClass_Homegrown(X,Y,training_to_test_ratio, 'iris')
disp('MATLAB SVM (Deliverable 3)')
Function_fitcecoc_01(X,Y,coding,training_to_test_ratio, 'iris')

disp(' ')
disp('---------------------------------------------------------------------------------------------------------- ')
disp('-----------------------------------------------------------------------------')
disp('  wine data')
myTable=readtable('wine_data.txt');
X=table2array( myTable(:,2:end));
Y=table2array(myTable(:,1));
disp('Homegrown (Deliverable 2)')
F_LinearNClass_Homegrown(X,Y,training_to_test_ratio, 'wine')
disp('MATLAB SVM (Deliverable 3)')
Function_fitcecoc_01(X,Y,coding,training_to_test_ratio, 'wine')
%%

disp(' ')
disp('---------------------------------------------------------------------------------------------------------- ')
disp('-----------------------------------------------------------------------------')
disp('  car data')
myTable=readtable('car_data.txt');
X = classreg.regr.modelutils.predictormatrix(myTable,'ResponseVar',size(myTable,2));
Y=table2array(myTable(:,7));
disp('Homegrown (Deliverable 2)')
F_LinearNClass_Homegrown(X,Y,training_to_test_ratio, 'car')
disp('MATLAB SVM (Deliverable 3)')
Function_fitcecoc_01(X,Y,coding,training_to_test_ratio, 'car')


%%
disp(' ')
disp('---------------------------------------------------------------------------------------------------------- ')
disp('-----------------------------------------------------------------------------')
disp('  ecoli data')
myTable=readtable('ecoli.txt');
X=table2array( myTable(:,2:8));
Y=table2array(myTable(:,9));
disp('Homegrown (Deliverable 2)')
F_LinearNClass_Homegrown(X,Y,training_to_test_ratio, 'ecoli')
disp('MATLAB SVM (Deliverable 3)')
Function_fitcecoc_01(X,Y,coding,training_to_test_ratio, 'ecoli')

function  [CE,values,W] =Function_createW(X,Y,lambda,nClasses)

for i_classes=1:nClasses
    y(1:length(Y))=0;y(Y==i_classes)=1;
    W(i_classes,:)=(X'*X+lambda(i_classes))\(X'*y');
end
[~,values]=max([X*W']');
CE=sum(Y~=values')/length(Y);

end

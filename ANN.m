clear all;
clc;

%******************DATASET GENERATION FOR SQUARE RECTANGLE****
rect_input1=randperm(900);
rect_input2=randperm(900);
rec_input=vertcat(rect_input1,rect_input2);
square_input=zeros(2,900);
trn_input=zeros(2,1800);
trn_input_m=zeros(2,1800);
trn_target=zeros(1,1800);
trn_target_m=zeros(1,1800);

for i=1:1:900
    if rec_input(1,i)==rec_input(2,i)
        rec_input(1,i)=rec_input(1,i)+1;
    end
end

for j=1:1:900
    square_input(1,j)=j;
    square_input(2,j)=j;
end
trn_input=horzcat(rec_input,square_input);

for j=1:1:1800
    if j<=900
        trn_target(1,j)=1;
    else 
        trn_target(1,j)=0;
    end
end

ix=randperm(1800);

for j=1:1:1800
    trn_input_m(:,j)=trn_input(:,ix(1,j));
    trn_target_m(:,j)=trn_target(:,ix(1,j));  
end

for j=1:1:1800
    if j<=1440
        trn_input_m(:,j)=trn_input_m(:,j);
        trn_target_m(:,j)=trn_target_m(:,j);
    else
        test_input_m(:,(j-1440))=trn_input_m(:,j);
        test_target_m(:,(j-1440))=trn_target_m(:,j);
    end
end
 
%***************TRAINING***************************
net = newff(minmax(trn_input_m),[10 1],{'logsig','purelin'},'trainlm');
net.trainParam.epochs = 300;
net.trainParam.goal = 0.0001;
input_weights_untrained=net.IW{1,1};
output_weights_untrianed=net.LW{2,1};

[net1,tr1]=train(net,trn_input_m,trn_target_m);

**********************OUTPUT******************************

output1=sim(net1,trn_input_m);
% %view(net1);
% %gensim(net1);
input_weights_trained=net1.IW{1,1};
output_weights_trianed=net1.LW{2,1};
net_bias=net1.b;
error=trn_target_m-output1;
perf=mse(error);

%*********************VALIDATION***************************
output3=sim(net1,test_input_m);
stem(output3);
hold on;
stem(test_target_m);

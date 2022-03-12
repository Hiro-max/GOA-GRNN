clear
clc
tic;
load data
y = data;

[bootstat, bootsam] = bootstrp(1000,@(x) Calc_MAE_GPR(x), y);

toc;
function MAE = Calc_MAE_MGOA(x)


% 对输入的数据处理
input = x(:,2:end);
output = x(:,1);


j = 1;
k = 1;
for i=1:size(input,1)

    if rem(i,3)==0
        
        p_test(j,:) = input(i,:);
        t_test(j,:) = output(i,:);
        j = j+1;
    else
        
        p_train(k,:) = input(i,:);
        t_train(k,:) = output(i,:);
        k = k+1;
    end
end

% [inputn,inputps] = mapminmax(p_train');
[outputn,outputps] = mapminmax(t_train');
% inputn_test = mapminmax('apply',p_test',inputps);
outputn_test = mapminmax('apply',t_test',outputps);

inputn = p_train';
% outputn = t_train;
inputn_test = p_test';
% outputn_test = t_test;

% *****************GOA***************************** 
N = 30;
max_iter = 30;
lb = 0.05;
ub = 2;
cMin = 0.00004;
cMax = 1;
dim = 1;

flag = 0;
if size(ub,1)==1
    ub = ones(dim,1)*ub;
    lb = ones(dim,1)*lb;
end

if(rem(dim,2)~=0)
    dim = dim+1;
    ub = [ub;ub];
    lb = [lb;-lb];
    flag = 1;
end



GrasshopperPositions1 = initialization_goa(N,dim,ub,lb);
%     GrasshopperPositions2 = (GrasshopperPositions1-min(GrasshopperPositions1))./(max(GrasshopperPositions1)-min(GrasshopperPositions1));

%增加Logistic映射
GrasshopperPositions=4*GrasshopperPositions1.*(1-GrasshopperPositions1);

for i=1:size(GrasshopperPositions,1)
    Tp=GrasshopperPositions(i,:)>ub';Tm=GrasshopperPositions(i,:)<lb';GrasshopperPositions(i,:)=(GrasshopperPositions(i,:).*(~(Tp+Tm)))+ub'.*Tp+lb'.*Tm;
end
GrasshopperFitness = zeros(1,N);
Sorted_grasshopper = zeros(10,2);
GrasshopperPositions_temp = zeros(10,2);

fitness_history = zeros(N,max_iter);
position_history = zeros(N,max_iter,dim);
%     Convergence_curve = zeros(1,max_iter);
Trajectories = zeros(N,max_iter);

for i=1:size(GrasshopperPositions,1)
    if flag==1
        GrasshopperFitness(1,i)=fit_fun(GrasshopperPositions(i,1:end-1),inputn',inputn_test',outputn',outputn_test');
    else
        GrasshopperFitness(1,i)=fit_fun(GrasshopperPositions(i,:),inputn',inputn_test',outputn',outputn_test');
    end

    fitness_history(i,1)=GrasshopperFitness(1,i);
    position_history(i,1,:)=GrasshopperPositions(i,:);
    Trajectories(:,1) = GrasshopperPositions(:,1);
end

[sorted_fitness,sorted_indexes] = sort(GrasshopperFitness);

for newindex=1:N
    Sorted_grasshopper(newindex,:)=GrasshopperPositions(sorted_indexes(newindex),:);
end

TargetPosition = Sorted_grasshopper(1,:);
TargetPosition_2 = Sorted_grasshopper(2,:);
TargetFitness = sorted_fitness(1);

% disp(['第一次迭代','最优值为',num2str(TargetFitness)])

l=2;
while l<max_iter+1

    c = cMax - (cMax-cMin)*(l/max_iter)^2;
    for i=1:size(GrasshopperPositions,1)
        temp = GrasshopperPositions';
        for k=1:2:dim
            S_i = zeros(2,1);
            for j=1:N
                if i~=j
                    Dist=distance(temp(k:k+1,j),temp(k:k+1,i));
                    r_ij_vec=(temp(k:k+1,j)-temp(k:k+1,i))/(Dist+eps);
                    xj_xi=2+rem(Dist,2);
                    s_ij=((ub(k:k+1)-lb(k:k+1))*c/2)*S_func(xj_xi).*r_ij_vec;
                    S_i=S_i+s_ij;
                end
            end
            S_i_total(k:k+1,:)=S_i;
        end
        mutation = rand;
        X_new = c*S_i_total'+(TargetPosition);

%             %加入变异
        r1 = round(mutation);
        if r1 == 0
            GrasshopperPositions_temp(i,:)=X_new'+(ub-X_new').*(1-mutation^((1-l/max_iter)^2));
        else 
            GrasshopperPositions_temp(i,:)=X_new'-(X_new'-lb).*(1-mutation^((1-l/max_iter)^2));
        end
    end

    GrasshopperPositions = GrasshopperPositions_temp;
    for i=1:size(GrasshopperPositions,1)
        Tp=GrasshopperPositions(i,:)>ub';Tm=GrasshopperPositions(i,:)<lb';GrasshopperPositions(i,:)=(GrasshopperPositions(i,:).*(~(Tp+Tm)))+ub'.*Tp+lb'.*Tm;
        if flag==1
            GrasshopperFitness(1,i)=fit_fun(GrasshopperPositions(i,1:end-1),inputn',inputn_test',outputn',outputn_test');
        else
            GrasshopperFitness(1,i)=fit_fun(GrasshopperPositions(i,:),inputn',inputn_test',outputn',outputn_test');
        end
        fitness_history(i,1)=GrasshopperFitness(1,i);
        position_history(i,1,:)=GrasshopperPositions(i,:);

        Trajectories(:,1)=GrasshopperPositions(:,1);

        if GrasshopperFitness(1,i)<TargetFitness
            TargetPosition = GrasshopperPositions(i,:);
            TargetFitness = GrasshopperFitness(1,i);
        end
    end
%     Convergence_curve(l) = TargetFitness;
        disp(['In iteration #',num2str(l),', target''s objective = ',num2str(TargetFitness)]);
    l = l+1;
end  

if(flag==1)
    TargetPosition = TargetPosition(1:dim-1);
end

% ***************************************************************************

%%
% GRNN网络预测
best_spread = TargetPosition;

Euclidean_D = distance_mat(inputn',inputn_test');
Guass_value = Guass(Euclidean_D,best_spread);
sum_mat = sum_layer(Guass_value,outputn');
output_mat_1 = output_layer(sum_mat);
%     output_mat = output_mat_1';
output_mat = mapminmax('reverse',output_mat_1',outputps);   

MAE = sum(abs(output_mat-t_test'),2)/size(t_test',2);
end


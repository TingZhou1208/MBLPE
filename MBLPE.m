function [ps,pf]=MBLPE(func_num,VRmin,VRmax,n_obj,Pop_Number,Max_gen,train_data)
global initial_flag
initial_flag=0;
% GPEA: Grey prediction evolution algorithm for global optimization
% Brief description: GPEA is a new optimization technique, the core of GPEA is considering the population series of evolutionary algorithms as a time series, and uses the even grey model as a reproduction operator to forecast the next population (without employing anymutation and crossover operators).
% Dimension: D --- dimension of solution space
%% Input:
%                             Dimension                   Description
%      D                      1 x 1                       dimension of solution space
%      Pop_Number             1 x 1                       population size
%      Max_Gen                1 x 1                       maximum  generations
%      VRmin                  1 x D                       low bound of variable in test function
%      VRmax                  1 x D                       up bound of variable in test function
%      func_num               1 x l                       the number of test function
%      th                     1 x 1                       difference threshold
%% Output:
%                             Dimension                   Description
%      bestFitness            1 x  Max_Gen                fitness values of the best individuals in each generation
%      bestFitness_gobal      1 x  1                      the fitness value of the gobal best individual
%      bestSolution_gobal     1 x  1                      the gobal best individual
%%  Reference and Contact
% Reference: [1]Zhongbo Hu, Xinlin Xu, and Qinghua Su et al., Grey prediction evolution algorithm for global optimization, Applied Mathematical Modelling, 2020, 79, 145–160.https://doi.org/10.1016/j.apm.2019.10.026
%            [2]Xinlin Xu, Zhongbo Hu, Qinghua Su, Yuanxiang Li, Juanhua Dai. Multivariable grey prediction evolution algorithm: A new metaheuristic, Applied Soft Computing, 2020, 89, 106086. https://doi.org/10.1016/j.asoc.2020.106086
%            [3]Canyun Dai, Zhongbo Hu, Zheng Li, Zenggang Xiong, Qinghua Su. An improved grey prediction evolution algorithm based on Topological Opposition-based learning. IEEE Access, vol. 8, pp. 30745-30762, 2020.https://doi.org/10.1109/ACCESS.2020.2973197
% Contact: For any questions, please feel free to send email to huzbdd@126.com.

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The initialization operator of GPEA.
%The initialization process of GPEA has to initialize three generation populations,it can be divided into two parts:
%The first part is to initialize the first generation with random uniform distribution;
%The second part is to generate the second and third generation population using DE.
% Note: The data sequences used by GPEA are all non-negative datasets.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize parameters
F=0.7;                    %Scaling factor or mutation factor,the value range is（0,1.2]
CR=0.7;                   %Crossover factor
Max_FES=Pop_Number*Max_gen;
%% Initialize the first generation population
n_var=size(VRmin,2);
Vrmin=ones(1,n_var).*(VRmin+abs(VRmin));                           %Initialize low bound of variable
Vrmax=ones(1,n_var).*(VRmax+abs(VRmin));                           %Initialize up bound of variable
Pop=1.*(repmat(Vrmin,Pop_Number,1)+(repmat(Vrmax,Pop_Number,1)-repmat(Vrmin,Pop_Number,1)).*rand(Pop_Number,n_var))>0.6;   %Initialize the first generation population
Pop=remove_empty(Pop,n_var);                                      %%移除空集
%% Evaluate the population
Fitness_Pop=zeros(Pop_Number,n_obj);
for i=1:Pop_Number
    Fitness_Pop(i,:)=feval(func_num,Pop(i,:),train_data);
end
fitcount=Pop_Number;                         %count the number of fitness evaluations
Parent=[Pop,Fitness_Pop];                    %put positions and velocities in one matrix
originPop(1)={Parent};                       %Store the first generation population
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for I=2:3
    %% Mutation operator
    %The non-equal three individuals randomly selected from the population to generate a mutation individual
    for i=1:Pop_Number
        Pop_list=randperm(Pop_Number);
        Pop_list(find(Pop_list==i))=[];
        Mutant(i,:)=Pop(Pop_list(1),:)+F*(Pop(Pop_list(2),:)-Pop(Pop_list(3),:));
    end
    for i=1:Pop_Number
        for j=1:n_var
            if Mutant(i,j)<Vrmin(j)||Mutant(i,j)>Vrmax(j)        %Make sure that individuals are in the setting bounds.
                Mutant(i,j)=Vrmin(j)+(Vrmax(j)-Vrmin(j))*rand;
            end
        end
    end
    %% Crossover operator
    %Intersect the target individual and its corresponding mutation individual to generate trial individual of the target individual.
    for i=1:Pop_Number
        for j=1:n_var
            r=randperm(n_var);
            if rand<=CR||j==r(1)
                trialPop(i,j)=Mutant(i,j);
            else
                trialPop(i,j)=Pop(i,j);
            end
        end
    end
      %% 后代的处理---类似于检测的作用，检测并修改
    trialPop=1.*(trialPop>0.6);
    trialPop=remove_empty(trialPop,n_var);
    trialPop=unique(trialPop,'rows','stable'); %%检查后代种群重复决策向量
    trialPop=handle_decision_vectors(Parent,trialPop,n_var); %%检查子代和父代是否有重复个体----处理重复的决策向量---提前处理为了减少不必要的计算量
    Pop_Number1=size(trialPop,1);
    %% Environment selection operator
    %Compare the value of the target individuals and trial individuals for population evaluation. Individuals with better fitness value will be selected to enter the next iteration
    Fitness_trial=zeros(Pop_Number1,n_obj);
    for i=1:Pop_Number1
        Fitness_trial(i,:)=feval(func_num,trialPop(i,:),train_data);    %Evaluate the trial population
        fitcount=fitcount+1;
    end
    Offspring=[trialPop Fitness_trial];
    Combin_Pop=[Offspring;Parent];
    Combin_Pop=unique(Combin_Pop,'rows','stable');       %%过滤掉相同的个体                                  %%删除重复个体
    Parent= EnvironmentalSelection(Combin_Pop,Pop_Number,n_var,n_obj);
    originPop(I)={Parent};                                                               %Store the second, third generation population
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% After a population initialization, the GPEA realizes the function-optimized process by looping a reproduction operator and a selection operator for updating the population.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for I=4:Max_gen
    %% The reproduction operator of GPEA.
    Pop_list1=randperm(Pop_Number);
    Pop_list2=randperm(Pop_Number);
    Pop_list3=randperm(Pop_Number);
    for k=1:Pop_Number
        for j=1:n_var
            X0=[originPop{1,1}(Pop_list1(k),j),originPop{1,2}(Pop_list2(k),j),originPop{1,3}(Pop_list3(k),j)];
            trialPop(k,j)=(4*X0(3)+X0(2)-2*X0(1))/3;                                %Linear fitting model
            %% Make sure that individuals are in the setting bounds.
            %If a newly generated individual exceeds the feasible region, the exceeding elements of the new individual are replaced by random numbers in the feasible region.
            if  trialPop(k,j)<Vrmin(j)||trialPop(k,j)>Vrmax(j)
                trialPop(k,j)=Vrmin(j)+(Vrmax(j)-Vrmin(j))*rand;
            end
        end
    end
      %% 后代的处理---类似于检测的作用，检测并修改
    trialPop=1.*(trialPop>0.6);
    trialPop=remove_empty(trialPop,n_var);
    trialPop=unique(trialPop,'rows','stable');
    Parent1=cell2mat(originPop');
    trialPop=handle_decision_vectors(Parent1,trialPop,n_var); %%检查子代和父代是否有重复个体
    Pop_Number1=size(trialPop,1);
    %% Environment selection operator
    Parent=originPop{1,3};
    Fitness_trial=zeros(Pop_Number1,n_obj);
    for i=1:Pop_Number1
        Fitness_trial(i,:)=feval(func_num,trialPop(i,:),train_data);    %Evaluate the trial population
        fitcount=fitcount+1;
    end
    Offspring=[trialPop Fitness_trial];
    Combin_Pop=[Offspring;Parent];
    Combin_Pop=unique(Combin_Pop,'rows','stable');%%过滤掉相同的个体                                     %%删除重复个体
    Parent= EnvironmentalSelection(Combin_Pop,Pop_Number,n_var,n_obj);
    originPop(1,4)={Parent};                          %Generate the true population
    originPop=originPop(2:4);                         %Update the population chain to produce offspring
    
    if fitcount>Max_FES
        break,
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Output bestFitness_gobal and bestSolution_gobal
ps=Parent(:,1:n_var);   %The fitness value of the gobal best individual
pf=Parent(:,n_var+1:n_var+n_obj);                %The gobal best individual
end
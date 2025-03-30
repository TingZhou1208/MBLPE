function [WE,rank] = fsFisher(feat,label)
    unlabel=unique(label,'rows','stable');
    numC = length(unlabel);
    [~, numF] = size(feat);
    out.W = zeros(1,numF);

    cIDX = cell(numC,1);
    n_i = zeros(numC,1);
    for j = 1:numC
        cIDX{j} = find(label(:)==unlabel(j));
        n_i(j) = length(cIDX{j});
    end

    for i = 1:numF
        temp1 = 0;
        temp2 = 0;
        f_i = feat(:,i); 
        u_i = mean(f_i); %%不同类，在相同特征下的均值

        for j = 1:numC
            u_cj = mean(f_i(cIDX{j})); %%同一个类，同一个特征下的均值
            var_cj = var(f_i(cIDX{j}),1); %%方差
            temp1 = temp1 + n_i(j) * (u_cj-u_i)^2; %%均值 同一个类均值与不同类均值之间的差异乘以类的数量
            temp2 = temp2 + n_i(j) * var_cj; %%方差
        end

        if temp1 == 0
            out.W(i) = 0;
        else
            if temp2 == 0
                out.W(i) = 100;
            else
                out.W(i) = temp1/temp2;
            end
        end
    end
[~, rank] = sort(out.W, 'descend');
%WE=out.W;
WE=(out.W-min(out.W))./(max(out.W)-min(out.W));



function [Sopt, nIterations, err_seq, obj_seq] = Algorithm2_AlternatingOptimization(H, PAPC, errtol, maxIterations)
P_sum = sum(PAPC);
%initialization
[~, nTx] = size(H);
[G,R] = qr(H);
q = ones(nTx,1);
err_seq = zeros(maxIterations,1);
obj_seq = zeros(maxIterations,1);
nIterations = maxIterations;
for iIter =1:maxIterations    
    % step 3
    [U_tilde, Sigma_tilde, V_tilde] = svd(R*diag(q.^(-1/2)),'econ');
    Sigma_tilde = diag(Sigma_tilde); 
    [Sigma_tilde, ind] = sort(Sigma_tilde,'descend'); % sort eigen channels for water filling
    U_tilde = U_tilde(:, ind); % rearrange the columns of U after sorting
    V_tilde = V_tilde(:,ind); % rearrange the columns of V after sorting

    zeroeigchan = (Sigma_tilde < 1e-7); % ignore very small eigen channels
    U_tilde(:,zeroeigchan) = []; % remove columns of U accordingly
    V_tilde(:,zeroeigchan) = []; % remove columns of V accordingly
    U = (G*U_tilde); % U from SVD of H*diag(q.^(-1))
    Sigma = Sigma_tilde.^2;
    power = wf(Sigma,P_sum); % perform water filling
    chan_pos = (power>0); % get stricly positive eigen channels
    power_pos = power(chan_pos);

    U(:,~chan_pos) = []; % remove columns of U that have no power
    Sigma = Sigma(chan_pos); % remove eigen channels that have no power
    S_bar = U*diag(power_pos)*U'; % the optimal S_bar for given Q
    % end of step 3

    % calculate the objective
    obj_seq(iIter) = real(log((det(diag(q) + H'*S_bar*H)))) - sum(log(q));
    if (iIter>1)
        err_seq(iIter-1)=abs(obj_seq(iIter)-obj_seq(iIter-1));
        if(err_seq(iIter-1)<errtol)
            err_seq(iIter:end) = [];
            obj_seq(iIter+1:end) = [];
            nIterations = iIter;
            break
        end
    end
    % step 6
    % Newton method for solving (26)
    V_dot = V_tilde(:,chan_pos);
    % step 5
    phi_inv = 1./q - real(diag(diag(q.^-0.5)*V_dot*diag(1./(1+1./(Sigma.*power_pos)))*(V_dot')*diag(q.^-0.5)));
    gamma = 0.01;
    fgamma=1;
    while(abs(fgamma) > errtol)
        fgamma = sum(1./(gamma+phi_inv./PAPC)) - P_sum;
        fgamma_diff = -sum(1./((phi_inv./PAPC + gamma).^2));
        gamma = gamma - fgamma/fgamma_diff;
    end
    q = 1./(phi_inv+ gamma*PAPC);
    % end of newton's method

    
end

[U, ~, V] = svd(H*diag(q.^(-1/2)), 'econ');
Sopt = diag(q.^(-1/2))*V*U'*S_bar*U*V'*diag(q.^(-1/2));
end

function power= wf(eigchan,P)
% water filling algorithm for solving the problem max sum(log(1+eigchan_i.*p_i)) s.t. sum(p) == P
% NOTE: eigen must be shorted in descending order
waterlevel = 1;
nEigchans = length(eigchan);
igamma = waterlevel./(eigchan);
temp = 0;
% water filling algorithm
for k = nEigchans:-1:1
    temp = (P+sum(1./eigchan(1:k))*waterlevel)/k;
    if ((temp-igamma(k))>0)
        break;
    end
end
power = max(temp-igamma,0);
end
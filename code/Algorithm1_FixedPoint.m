function [Sopt, nIterations, err_seq,obj_seq] = Algorithm1_FixedPoint(H, PAPC, errtol, maxIterations)
%initialization
[m,n] = size(H);

lambda_tilde = (ones(n,1)); 
lambda = 1./lambda_tilde;
[~, R] = qr(H); % step 2
err_seq = zeros(maxIterations,1);
obj_seq = zeros(maxIterations,1);
nIterations = maxIterations;
for iIter=1:maxIterations    
    % step 4
    [~, Sigma_bar, V] = svd(R*diag(lambda.^(-1/2)));
    Sigma_bar(Sigma_bar<0) = 0;
    Sv = Sigma_bar(Sigma_bar>0).^(-2);
    Svs = 1-Sv;
    Svs(Svs <0) = 0;
    % step 6
    Phi = V*diag(1-[Svs;zeros(n-length(Svs),1)])*V';
    % step 7
    S = diag((lambda).^(-1)) - diag((lambda).^(-1/2))*Phi*diag((lambda).^(-1/2));
    obj_seq(iIter) = real(log(det(eye(m)+H*S*H')));
    % step 8
    err_seq(iIter) = abs(sum((((lambda))).*(diag(S)-PAPC)));
    % step 9
    lambda_tilde = real(PAPC + diag(Phi).*lambda_tilde);
    % step 10
    lambda = ((lambda_tilde).^(-1));
    
    %break in case exceeding the limit
    if (err_seq(iIter) < errtol)
        err_seq(iIter+1:maxIterations)=[];
        obj_seq(iIter+1:maxIterations)=[];
        nIterations = iIter;
        break;
    end
    
    
end

Sopt = S;


end
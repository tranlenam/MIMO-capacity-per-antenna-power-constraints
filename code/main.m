clear ;
clc
rng(1)
%initialization
SNRdB = 0;
P = 10.^(SNRdB/10);
eps = 1e-6;

maxIterations = 50;


% nTx = 5;
% nRx = 3;
% H = (randn(nRx, nTx) + 1i*randn(nRx, nTx))/sqrt(2);

H =[0.2581+1i*0.6535i 0.2623+1i*0.9434i;
0.4385+1i*0.3081 0.4090-1i*0.2288];
[nRx,nTx]=size(H);

PAPC = (P/nTx)*ones(nTx,1); % equal power constraint
% cvx_solver mosek
cvx_expert true
cvx_begin quiet
variable X(nTx,nTx) complex semidefinite
maximize(log_det(eye(nRx)+H*X*H'))
diag(X) <= PAPC
X == hermitian_semidefinite(nTx)
cvx_end
cvx_optval

%Alg1, fixed point
[Sopt_fp, nIterations_fp, err_seq_fp,obj_seq_fp] = Algorithm1_FixedPoint(H, PAPC, eps, maxIterations);

CMIMO_Alg1 = real(log(det(eye(nRx) + H*Sopt_fp*H')))

%Alg2, alternating optimization
[Sopt_ao, nIterations_ao, err_seq_ao,obj_seq_ao] = Algorithm2_AlternatingOptimization(H, PAPC, eps, maxIterations);
CMIMO_Alg2 = real(log(det(eye(nRx) + H*Sopt_ao*H')))



%plot duality
subplot(2,1,1)
semilogy(1:nIterations_fp,err_seq_fp,'--b','LineWidth',1.5);
hold on
semilogy(1:nIterations_ao-1,err_seq_ao,'-k','LineWidth',1.5);
legend('Algorithm 1', 'Algorithm 2','Location','Best');
xlabel('Iteration Index','FontSize',12,'FontWeight','bold');
ylabel('Duality gap','FontSize',12,'FontWeight','bold');
title('Residual error')
subplot(2,1,2)
plot(obj_seq_fp,'--b')
hold on
plot(obj_seq_ao,'-k')
plot(cvx_optval*ones(length(obj_seq_ao),1),'-r')
title('Convergence of the objective')
legend('Algorithm 1', 'Algorithm 2','Optimal Objective (CVX)','Location','Best');
saveas(gcf,'../results/convergence.png')



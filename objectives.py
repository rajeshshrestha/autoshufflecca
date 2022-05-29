import torch


class cca_loss():
    def __init__(self, k_eigen_check_num, use_all_singular_values, device):
        self.k_eigen_check_num = k_eigen_check_num
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def loss(self, H1, H2, M):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9
        lambda_M = 0

        H1, H2 = H1.t(), H2.t()


        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)


        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)

        # print(H1.shape, H2.shape)

        # print(torch.matrix_rank(SigmaHat11))
        # print(torch.matrix_rank(SigmaHat22))

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11+1e-9*torch.randn_like(SigmaHat11), eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22+1e-9*torch.randn_like(SigmaHat11), eigenvectors=True)

        # print(D1, V1)
        # print(D2, V2)
        # exit()


        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        

        assert self.k_eigen_check_num<=m , "High number of eigen value checked than the number of output space"
        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
        U, V = torch.symeig(trace_TT, eigenvectors=True)
        U = torch.where(U>eps, U, (torch.ones(U.shape).double()*eps).to(self.device))
        if not self.use_all_singular_values:
            # just the top self.outdim_size singular values are used
            U = U.topk(self.k_eigen_check_num)[0]
        corr = torch.sum(torch.sqrt(U))

        M_reg = torch.sum(torch.sum(torch.sum(torch.abs(M), dim=2) - (torch.sum(M**2, dim=2))**0.5,dim=1)) +\
            torch.sum(torch.sum(torch.sum(torch.abs(M), dim=1) - (torch.sum(M**2, dim=1))**0.5,dim=1))

        
        
        return -corr + lambda_M/m * M_reg

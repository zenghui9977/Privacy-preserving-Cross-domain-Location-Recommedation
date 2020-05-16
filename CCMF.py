import torch
from torch.autograd import Variable

class CCMF:
    def __init__(self, w_a, w_t, learning_rate, lamda_Q, lamda_P, k, auxiliary_user_num, target_user_num, item_num):
        self.P_a = Variable(torch.randn([auxiliary_user_num, k]), requires_grad=True)
        self.P_t = Variable(torch.randn([target_user_num, k]), requires_grad=True)
        self.Q = Variable(torch.randn([item_num, k]), requires_grad=True)
        self.w_a = w_a
        self.w_t = w_t
        self.learning_rate = learning_rate
        self.lamda_Q = lamda_Q
        self.lamda_P = lamda_P
        self.k = k

    '''
    auxiliary_data: shape [auxiliary_user_num * item_num]
    target_data: shape [target_user_num * item_num]
    confidece_matrix: shape [auxiliary_user_num * item_num]
    '''
    def update(self,auxiliary_data, target_data, confidence_matrix):

        Pred_a = torch.mm(self.P_a, self.Q.t())
        Pred_t = torch.mm(self.P_t, self.Q.t())

        # epsilon_a: shape[auxiliary_user_num * item_num]
        # epsilon_t: shape[target_user_num * item_num]
        epsilon_a = 2 * (auxiliary_data - Pred_a)
        epsilon_t = 2 * (target_data - Pred_t)

        # auxiliary domain
        # w_a * confidence_matrix * epsilon_a * Q - lamda_P * P_a
        # update_p_a_ : shape [auxiliary_user_num * item_num]
        # update_p_a : shape [auxiliary_user_num, k]
        update_p_a_ = self.w_a * torch.mul(torch.Tensor(confidence_matrix), epsilon_a)
        update_p_a = torch.mm(update_p_a_, self.Q) - self.lamda_P * self.P_a

        # target domain
        # w_t * epsilon_t * Q - lanmda_P * P_t
        # update_p_t = shape [target_user_num, k]
        update_p_t = self.w_t * torch.mm(epsilon_t , self.Q) - self.lamda_P * self.P_t

        # update_q : shape [item_num, k]
        update_q = torch.mm(update_p_a_.t() , self.P_a) + self.w_t * torch.mm(epsilon_t.t(), self.P_t) - 2 * self.lamda_Q * self.Q


        # all matrix update
        self.P_a = self.P_a + self.learning_rate * update_p_a
        self.P_t = self.P_t + self.learning_rate * update_p_t
        self.Q = self.Q + self.learning_rate * update_q

    # predict target domain
    def predict(self):
        prediction = torch.mm(self.P_t, self.Q.t())
        return torch.sigmoid(prediction)






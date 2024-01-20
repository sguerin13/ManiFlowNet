import torch
import pytorch_lightning as pl
###############################
#             CODE            #
###############################

class MAPE(pl.LightningModule):
    def __init__(self):
        super(MAPE,self).__init__()

    def forward(self,output,target):

        loss = torch.mean((output - target).abs()/(target+.00001).abs())
        return loss

class MaskedMAPE(pl.LightningModule):
    def __init__(self,mode='mean'):
        super(MaskedMAPE,self).__init__()
        self.mode = mode

    def forward(self,output,target):

        loss = (output - target).abs()/(target+.00001).abs()
        mask = torch.ones(target.shape).cuda()

        for i in range(target.shape[2]):
            if (target[0,:,i,0].sum()==0):
                mask[0,:,i,0] = 0
        loss = loss*mask
        
        if self.mode == 'mean':
            loss = torch.mean(loss)

        elif self.mode == 'sum':
            loss = torch.sum(loss)

        if loss < 0:
            raise Exception("Negative Loss")

        return loss

class MaskedMSE(pl.LightningModule):
    def __init__(self,mode = 'mean'):
        super(MaskedMSE,self).__init__()
        self.mode = mode

    def forward(self,output,target):
        loss = torch.nn.MSELoss(reduction='none')(output,target)
        mask = torch.ones(target.shape).cuda()
        
        for i in range(target.shape[2]):
            if (target[0,:,i,0].sum()==0):
                mask[0,:,i,0] = 0
        loss = loss*mask
        
        if self.mode == 'mean':
            loss = torch.mean(loss)

        elif self.mode == 'sum':
            loss = torch.sum(loss)

        return loss

###############################
#            TESTS            #
###############################

def test_masked_mse_loss():
    target_octree_tensor = torch.tensor([[[
                                          [1,0],
                                          [1,0],
                                          [1,0],
                                          [1,0]
                                        ]]]).float()

    output_octree_tensor = torch.tensor([[[
                                          [2,1],
                                          [2,1],
                                          [2,1],
                                          [2,1]
                                        ]]]).float()
    
    target_octree_tensor = target_octree_tensor.reshape([1,4,2,1])
    output_octree_tensor = output_octree_tensor.reshape([1,4,2,1])
    assert target_octree_tensor.shape == (1,4,2,1)

    masked_mse = MaskedMSE(mode='sum')(output_octree_tensor,target_octree_tensor)
    assert masked_mse == torch.tensor(4)



if __name__ == '__main__':
    test_masked_mse_loss()

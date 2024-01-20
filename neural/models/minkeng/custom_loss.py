import torch
import pytorch_lightning as pl
from torchmetrics import MeanAbsolutePercentageError

###############################
#             CODE            #
###############################

class MaxValWAPE(pl.LightningModule):
    '''
    Max Value Weighted Average Percent Error:
     - Weight percent error by the max(abs(gt),abs(pred))

     sum(max(a_i,b_i)*(abs(a_i-b_i)/(abs(a_i)))
     ------------------------------------------
                  sum(max(a_i,b_i))
    '''

    def __init__(self):
        super(MaxValWAPE,self).__init__()

    def forward(self,output,target):
        output = output.flatten()
        target = target.flatten()
        max_vals = torch.maximum(torch.abs(output), torch.abs(target))
        delta = (output-target).abs()/(torch.abs(target))
        loss = torch.sum(max_vals*delta)/torch.sum(max_vals)
        return loss


class MaskedMaxValWAPE(pl.LightningModule):
    '''
    Max Value Weighted Average Percent Error:
     - Weight percent error by the max(abs(gt),abs(pred))

     sum(max(a_i,b_i)*(abs(a_i-b_i)/(abs(a_i)))
     ------------------------------------------
                  sum(max(a_i,b_i))
    '''

    def __init__(self):
        super(MaskedMaxValWAPE,self).__init__()

    def forward(self,output,target):
        mask = torch.zeros(target.shape[0],4).cuda() #[n_points,dim]
        for i in range(target.shape[0]):
            if target[i,-1] > 0:
                mask[i,:] = 1  # inlet or outlet
        mask = mask.flatten()
        output = output.flatten()
        target = target[:,:4].flatten()
        max_vals = torch.maximum(torch.abs(output), torch.abs(target))
        delta =((output-target)*mask).abs()/(torch.abs(target) + .001)
        loss = torch.sum(max_vals*delta)/torch.sum(max_vals)
        return loss


class MaskedMAPE(pl.LightningModule):
    '''
    output shape : [:,4]
    target shape : [:,5] last value in D2 is the mask value
    '''
    def __init__(self,mode='mean'):
        super(MaskedMAPE,self).__init__()
        self.mode = mode

    def forward(self,output,target):
        mask = torch.zeros(target.shape[0],4).cuda() #[n_points,dim]
        for i in range(target.shape[0]):
            if target[i,-1] > 0:
                mask[i,:] = 1  # inlet or outlet
        mask = mask.flatten()
        output = output.flatten()
        target = target[:,:4].flatten()
        loss = ((output - target)*mask).abs()/((target).abs() +.001)
        loss = torch.sum(loss)/torch.sum(mask)
        # # mask = torch.ones(target.shape).cuda()

        # # for i in range(target.shape[2]):
        # #     if (target[0,:,i,0].sum()==0):
        # #         mask[0,:,i,0] = 0
        # # loss = loss*mask
        
        # if self.mode == 'mean':
        #     loss = torch.mean(loss)

        # elif self.mode == 'sum':
        #     loss = torch.sum(loss)

        if loss < 0:
            raise Exception("Negative Loss")

        return loss


class MaskedWAPE(pl.LightningModule):
    '''
    weighted average percent error
        - since some near-edge values are 0 or are close to zero. this blows up the
          av. error, therefore going to weight it by the value of the target value.

        output shape : [:,4]
        target shape : [:,5] last value in D2 is the mask value
    '''
    def __init__(self,mode='mean'):
        super(MaskedWAPE,self).__init__()
        self.mode = mode

    def forward(self,output,target):

        mask = torch.zeros(target.shape[0],4).cuda() #[n_points,dim]
        for i in range(target.shape[0]):
            if target[i,-1] > 0:
                mask[i,:] = 1  # inlet or outlet
        mask = mask.flatten()
        output = output.flatten()
        target = target[:,:4].flatten()

        delta = ((output - target)*mask).abs()
        target_vals = (target*mask).abs()
        zero_offset = torch.ones(target_vals.shape).cuda()*.00001
        zero_offset = zero_offset*mask # only add the offset to values we're using

        target_with_offset = target_vals + zero_offset
        loss = torch.sum(delta)/torch.sum(target_with_offset)
       
        if loss < 0:
            raise Exception("Negative Loss")

        return loss


class MaskedMSE(pl.LightningModule):
    def __init__(self,mode = 'mean'):
        super(MaskedMSE,self).__init__()
        self.mode = mode

    def forward(self,output,target):
        '''
        loss is masking based on if the node is an inlet or outlet
        the feature to include in loss is the last feature tacked on

        output shape : [:,4]
        target shape : [:,5] last value in D2 is the mask value
        '''
        loss = torch.nn.MSELoss(reduction='none')(output,target[:,:4])
        mask = torch.zeros(target.shape[0],4).cuda() #[n_points,dim]
        for i in range(target.shape[0]):
            if target[i,-1] > 0:
                mask[i,:] = 1  # inlet or outlet
        
        loss = loss[:,:4]*mask #drop the last column    
        if self.mode == 'mean':
            loss = torch.sum(loss)/torch.sum(mask)

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)#(256,256)
        hardnum = self.opt.hardnum #2
        mm_a = (torch.arange(scores.size(0))//self.opt.hardnum +1)*self.opt.hardnum
        mask_a = torch.arange(im.size(0)).view(im.size(0),1).expand_as(scores)  #(256,256)
        mask1 = (mask_a<mm_a.long())  #(256,256)
        mask = mask1 * mask1.t()  #(256,256)
        if torch.cuda.is_available():
            I = mask.cuda()

        #caption retrieval 
        scores_inner = torch.masked_select(scores,I).reshape(scores.size(0)//hardnum, hardnum, hardnum) #(128,2,2)
        
        scores_image = scores_inner.min(dim=2)[0].reshape((-1,1)) #(256,1)
        cost_s = (self.margin + scores - scores_image.view(im.size(0),1).expand_as(scores)).clamp(min=0) #(256,256)

        #image retrieval
        scores_caption = scores_inner.min(dim=1)[0].reshape((1,-1)) #(1,256)
        cost_im = (self.margin + scores - scores_caption.view(1,s.size(0)).expand_as(scores)).clamp(min=0)

        
        cost_s = cost_s.masked_fill_(I, 0) #(256,256)
        cost_im = cost_im.masked_fill_(I, 0) #(256,256)

        if self.max_violation:
            cost_im = cost_im.max(0)[0]
            cost_s = cost_s.max(1)[0]
            
        cost_im =cost_im.sum()    
        cost_s =cost_s.sum()

        return cost_im, cost_s

def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities



class TransformerContrastiveLoss(nn.Module):
    """
    Compute contrastive loss combined with hard negative triplet loss for the Transformer module.
    """

    def __init__(self, batch_size, margin=1.0, temperature=0.6, num_hard_negatives=1, lambda_contrastive=0.5):
        super(TransformerContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.margin = margin  # Margin for triplet loss
        self.num_hard_negatives = num_hard_negatives  # Number of hard negatives to select for triplet loss
        self.lambda_contrastive = lambda_contrastive  # Weighting factor for contrastive loss
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer("negatives_mask",
                             (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(self.device)).float())

    def forward(self, img_emb, cap_emb, labels):
        # Normalize image and text embeddings
        z_i = F.normalize(img_emb, dim=1)  # (256,1024)
        z_j = F.normalize(cap_emb, dim=1)  # (256,1024)

        # Concatenate the embeddings
        representations = torch.cat([z_i, z_j], dim=0)  # (512,1024)

        # Calculate similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # (512,512)

        # Expand negatives mask
        negatives_mask_expanded = torch.cat([self.negatives_mask, self.negatives_mask], dim=0)  # (512,512)
        negatives_mask_expanded = torch.cat([negatives_mask_expanded, negatives_mask_expanded], dim=1)  # (512,512)

        # Extract positive pairs similarities
        sim_ij = torch.diag(similarity_matrix, self.batch_size)[:self.batch_size]  # (128,)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)[:self.batch_size]  # (128,)
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # (256,)

        # Calculate positive pairs loss with weights (for contrastive loss)
        nominator_expanded = torch.cat([torch.exp(positives / self.temperature)] * 2)  # (512,)

        # Select hard negatives for triplet loss
        hard_negatives = similarity_matrix * negatives_mask_expanded  # (512,512)
        hard_negatives_values, hard_negatives_indices = torch.topk(hard_negatives, k=self.num_hard_negatives, dim=1,
                                                                   largest=True, sorted=False)  # (512,1)

        # Calculate negative pairs loss (for contrastive loss)
        denominator = torch.sum(torch.exp(hard_negatives_values / self.temperature), dim=1)  # (512,)

        # Calculate contrastive loss
        contrastive_loss = torch.sum(-torch.log(nominator_expanded / denominator)) / (2 * self.batch_size)

        # Calculate triplet loss with hard negatives
        hard_neg_emb = representations[hard_negatives_indices.view(-1)]  # (512,1024)

        # Expand z_i to match the size of hard_neg_emb using repeat
        z_i_expanded = z_i.repeat(2, 1)  # (512, 1024)

        # Reshape z_i_expanded to match hard_neg_emb
        z_i_expanded = z_i_expanded.view(512, self.num_hard_negatives, z_i.size(1))  # (512, 1, 1024)

        # Calculate pairwise distances
        pos_dist = F.pairwise_distance(z_i_expanded.squeeze(1), z_j.repeat(2, 1))  # (512,)
        neg_dist = torch.min(F.pairwise_distance(z_i_expanded, hard_neg_emb), dim=1)[0]  # Use the minimum distance from the hard negatives

        # Compute triplet loss
        triplet_loss = F.relu(pos_dist - neg_dist + self.margin)
        triplet_loss = triplet_loss.mean()

        # Combine contrastive loss and triplet loss
        total_loss = triplet_loss + self.lambda_contrastive * contrastive_loss

        return total_loss




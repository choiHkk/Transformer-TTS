from torch import nn


class TransformerTTSLoss(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/loss_function.py"""
    def __init__(self, r_gate=5.):
        super(TransformerTTSLoss, self).__init__()
        """
        Neural Speech Synthesis with Transformer Network
        
        page 4 : 3.7 Mel Linear, Stop Linear and Post-net
            It’s worth mentioning that, for the stop linear, there is only one positive sample
            in the end of each sequence which means ”stop”, while hundreds of negative samples for other frames. 
            This imbalance may result in unstoppable inference.
            We impose a positive weight (5.0 ∼ 8.0) on the tail positive stop token when calculating binary cross entropy loss, 
            and this problem was efficiently solved.
        """
        self.r_gate = r_gate

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out_postnet, mel_out, gate_out = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target) * self.r_gate
        return mel_loss + gate_loss
    
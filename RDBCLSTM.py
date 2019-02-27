import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self,  input_chans, filter_size, growthrate):
        super(CLSTM_cell, self).__init__()

        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = growthrate
        self.growthrate = growthrate
        self.efficient = True
        # self.batch_size=batch_size
        self.padding = (filter_size - 1) / 2  # in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, prev_features, hidden_state):
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            # bottleneck_output = bn_function(prev_features,hidden_state,self.growthrate)
            bottleneck_output = checkpoint(self.bn_function,prev_features,hidden_state[0],hidden_state[1])
        else:
            bottleneck_output = self.bn_function(prev_features,hidden_state[0],hidden_state[1])

        return bottleneck_output

    def init_hidden(self, batch_size,shape):
        return (Variable(torch.zeros(batch_size, self.num_features, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, self.num_features, shape[0], shape[1])).cuda())

    def bn_function(self,inputs, hidden_state1,hidden_state2):

        concated_features = inputs

        hidden, c = hidden_state1, hidden_state2 # hidden and c are images with several channels
        # print 'hidden ',hidden.size()
        # print 'input ',input.size()
        combined = torch.cat((concated_features, hidden), 1)  # oncatenate in the channels
        # print 'combined',combined.size()
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.growthrate, dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c


## This is RDC-LSTM **************
class CLSTM(nn.Module):


    def __init__(self, input_chans, filter_size,growthrate, num_layers):
        super(CLSTM, self).__init__()

        self.input_chans = input_chans
        self.filter_size = filter_size
        self.growthrate = growthrate

        self.num_layers = num_layers
        cell_list = []
        Invcell_list = []
        cell_list.append(
            CLSTM_cell(self.input_chans, self.filter_size, self.growthrate).cuda())  # the first

        Invcell_list.append(CLSTM_cell( self.input_chans, self.filter_size, self.growthrate).cuda())

        for idcell in xrange(1, self.num_layers):
            cell_list.append(CLSTM_cell( 2* (growthrate + (idcell-1) * growthrate), self.filter_size, self.growthrate).cuda())

        for idcell in xrange(1, self.num_layers):
            Invcell_list.append(CLSTM_cell(2 *(growthrate +  (idcell-1) * growthrate), self.filter_size, self.growthrate).cuda())

        self.cell_list = nn.ModuleList(cell_list)
        self.Invcell_list = nn.ModuleList(Invcell_list)
        self.apply(weights_init)

    def forward(self, input):

        current_input = input
        seq_len, B, C, H, W = current_input.size()

        next_hidden = []  # hidden states(h and c)
        output = current_input
        longrangeout = []
        for idlayer in xrange(self.num_layers):  # loop for every layer

            hidden_c = self.cell_list[idlayer].init_hidden(B, (H, W))  # hidden and c are images with several channels

            hidden_c_b = hidden_c
            if self.longrange:
                hidden_c = self.longrange[idlayer]


            all_output = []
            output_inner = []
            output_inner_b = []
            output_inner_f = []
            for t in xrange(seq_len):  # loop for every step
                hidden_c = self.cell_list[idlayer](output[t, ...],
                                                   hidden_c)  # cell_list is a list with different conv_lstms 1 for every layer
                output_inner_f.append(hidden_c[0])

                if t == seq_len-3:
                    longrangeout.append(hidden_c)

            for b in xrange(seq_len):
                hidden_c_b = self.Invcell_list[idlayer](output[seq_len-b-1, ...],
                                                        hidden_c_b)
                output_inner_b.append(hidden_c_b[0])

            for h in range(len(output_inner_f)):
                v1 = output_inner_f[h]
                v2 = output_inner_b[seq_len-1-h]
                v3 = torch.cat((v1,v2),1)
                output_inner.append(v3)

            #next_hidden.append(hidden_c+hidden_c_b)
            current_input = torch.cat(output_inner, 0).view(output.size(0),
                                                            *output_inner[0].size())  # seq_len,B,chans,H,W

            if idlayer == 0:
                output = current_input
            else:
                output = torch.cat((output,current_input), 2)

        return  output + input,longrangeout
        #return current_input

    def setlongrange(self,longrange):
        self.longrange = longrange

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states




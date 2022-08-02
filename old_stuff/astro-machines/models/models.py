import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv1d(int(in_planes), int(out_planes), kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def predict(in_planes):
    return nn.Conv1d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2):
    return nn.Sequential(
        nn.ConvTranspose1d(int(in_planes), int(out_planes), kernel_size=kernel_size, stride=stride, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def in_channels(layer):
    return layer._modules['0'].in_channels

class ae1d(nn.Module):

    def __init__(self):
        super(ae1d,self).__init__()

        self.conv1   = conv(1,16, kernel_size=7, stride=2)
        self.conv2   = conv(16,16, kernel_size=5, stride=2)
        self.conv3   = conv(16,16, kernel_size=5, stride=2)
        self.conv4   = conv(16,32, stride=2)
        self.conv5   = conv(32,32, stride=2)
        self.conv6   = conv(32,32, stride=2)
        self.conv7   = conv(32,64, stride=2)
        self.conv8   = conv(64,64, stride=2)
        self.conv9   = conv(64,64, stride=2)
        self.conv10   = conv(64,128, stride=2)
        self.conv11   = conv(128,128, stride=2)
        self.conv12   = conv(128,128, stride=2)
        self.conv13   = conv(128,256, stride=2)
        self.conv14   = conv(256,512, stride=2)
        self.conv15 = conv(512, 512)

        def estimate_size(self,shape):
            x = torch.zeros((1,shape[0],shape[1]))
            
            out_conv1 = self.conv1(x)
            out_conv2 = self.conv2(out_conv1)
            out_conv3 = self.conv3(out_conv2)
            out_conv4 = self.conv4(out_conv3)
            out_conv5 = self.conv5(out_conv4)
            out_conv6 = self.conv6(out_conv5)
            out_conv7 = self.conv7(out_conv6)
            out_conv8 = self.conv8(out_conv7)
            out_conv9 = self.conv9(out_conv8)
            out_conv10 = self.conv10(out_conv9)
            out_conv11 = self.conv11(out_conv10)
            out_conv12 = self.conv12(out_conv11)
            out_conv13 = self.conv13(out_conv12)
            out_conv14 = self.conv14(out_conv13)
            out_conv15 = self.conv15(out_conv14)

            out_conv15_flat = out_conv15.view(out_conv15.size(0),-1)
            return out_conv15.size(),out_conv15_flat.size()

        out15_shape,out15_shape_flat = estimate_size(self,(1, 327680))
            
        self.n_bottleneck_input_elements = out15_shape_flat[1]
        self.fc11 = nn.Linear(self.n_bottleneck_input_elements,128)
        self.fc12 = nn.Linear(self.n_bottleneck_input_elements,128)

        self.n_bottleneck_output_elements = self.n_bottleneck_input_elements
        self.fc2 = nn.Linear(128,self.n_bottleneck_output_elements)
                           
        self.deconv15 = deconv(512,512,kernel_size=3,stride=1)
        self.deconv14 = deconv(512,256) 
        self.deconv13 = deconv(256,128) 
        self.deconv12 = deconv(128,128) 
        self.deconv11 = deconv(128,128) 
        self.deconv10 = deconv(128,64) 
        self.deconv9 = deconv(64,64) 
        self.deconv8 = deconv(64,64) 
        self.deconv7 = deconv(64,32) 
        self.deconv6 = deconv(32,32) 
        self.deconv5 = deconv(32,32) 
        self.deconv4 = deconv(32,16) 
        self.deconv3 = deconv(16,16) 
        self.deconv2 = deconv(16,16) 
        self.deconv1 = deconv(16,1) 
        self.predict0 = predict(1)


    def encode(self,x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        out_conv8 = self.conv8(out_conv7)
        out_conv9 = self.conv9(out_conv8)
        out_conv10 = self.conv10(out_conv9)
        out_conv11 = self.conv11(out_conv10)
        out_conv12 = self.conv12(out_conv11)
        out_conv13 = self.conv13(out_conv12)
        out_conv14 = self.conv14(out_conv13)
        out_conv15 = self.conv15(out_conv14)

        out_conv15_flat = out_conv15.view(out_conv15.size(0),-1)

        mu = self.fc11(out_conv15_flat)
        logvar = self.fc12(out_conv15_flat) 
        return mu
    
    def decode(self,latent_vector):
        latent_to_basis = self.fc2(latent_vector)
        latent_to_basis_view = latent_to_basis.view(latent_to_basis.size(0),512,-1)

        out_deconv15 = self.deconv15(latent_to_basis_view)
        out_deconv14 = self.deconv14(out_deconv15)
        out_deconv13 = self.deconv13(out_deconv14)
        out_deconv12 = self.deconv12(out_deconv13)
        out_deconv11 = self.deconv11(out_deconv12)
        out_deconv10 = self.deconv10(out_deconv11)
        out_deconv9 = self.deconv9(out_deconv10)
        out_deconv8 = self.deconv8(out_deconv9)
        out_deconv7 = self.deconv7(out_deconv8)
        out_deconv6 = self.deconv6(out_deconv7)
        out_deconv5 = self.deconv5(out_deconv6)
        out_deconv4 = self.deconv4(out_deconv5)
        out_deconv3 = self.deconv3(out_deconv4)
        out_deconv2 = self.deconv2(out_deconv3)
        out_deconv1 = self.deconv1(out_deconv2)
        out0       = self.predict0(out_deconv1)

        return out0
        
    def forward(self, x):
        latent_vector = self.encode(x)
        out0 = self.decode(latent_vector)
        return out0

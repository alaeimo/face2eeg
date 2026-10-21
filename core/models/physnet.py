# face2eeg/core/models/physnet.py

"""
PhysNet 3D-CNN encoder-decoder for EEG prediction from facial video.
"""
import torch.nn as nn

class PhysNet_3DCNN_ED(nn.Module):
    def __init__(self, video_length=50, signal_length=128):
        super(PhysNet_3DCNN_ED, self).__init__()
        self.y = signal_length
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)


        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)


        # self.poolspa = nn.AdaptiveMaxPool3d((signal_length,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((signal_length,1,1))

    def forward(self, x):	    	# x [3, T, W, H]
        [batch,channel,length,width,height] = x.shape		     # x [-1,3, T, W, H]

        x = self.ConvBlock1(x)		     # x [16, T, W, H]
        x = self.MaxpoolSpa(x)       # x [16, T,W/2, H/2]

        x = self.ConvBlock2(x)		    # x [32, T, W/2, H/2]
        x = self.ConvBlock3(x)	    	# x [64, T, W/2, H/2]
        x = self.MaxpoolSpaTem(x)      # x [64, T/2, W/4, H/4]    Temporal halve

        x = self.ConvBlock4(x)		    # x [64, T/2, W/4, H/4]
        x = self.ConvBlock5(x)	    	# x [64, T/2, W/4, H/4]
        x = self.MaxpoolSpaTem(x)      # x [64, T/4, W/8, H/8]


        x = self.ConvBlock6(x)		    # x [64, T/4, W/8, H/8]
        x = self.ConvBlock7(x)	    	# x [64, T/4, W/8, H/8]
        x = self.MaxpoolSpa(x)      # x [64, T/4, W/16, H/16]

        x = self.ConvBlock8(x)		    # x [64, T/4, W/16, H/16]
        x = self.ConvBlock9(x)		    # x [64, T/4, W/16, H/16]


        x = self.upsample(x)		    # x [64, T/2, W/16, H/16]
        x = self.upsample2(x)		    # x [64, T, W/16, H/16]


        x = self.poolspa(x)     # x [64, Signal_Length, 1,1]
        x = self.ConvBlock10(x)    # x [1, Signal_Length, 1,1]
        EEG = x.view(-1,self.y)    #Signal_Length
        return EEG


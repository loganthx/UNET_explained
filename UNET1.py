import torch, torch.nn as nn, torch.nn.functional as F

# INPUT_SHAPE (3, 48, 48)
class UNET(nn.Module):
	def __init__(self):
		super().__init__()
		self.pool = nn.MaxPool2d(2)

		self.up_block1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, padding=1), # OUT (64, 48, 48)
			nn.Conv2d(64, 64, 3, padding=1), # OUT (64, 48, 48) 
			)
		# MAXPOOL((64, 48, 48)) = (64, 24, 24)

		self.up_block2 = nn.Sequential(
			nn.Conv2d(64, 128, 3, padding=1), # OUT (128, 24, 24)
			nn.Conv2d(128, 128, 3, padding=1), # OUT (128, 24, 24) 
			)
		# MAXPOOL((128, 24, 24)) = (128, 12, 12)

		self.bottleneck_conv = nn.Conv2d(128, 256, 3, padding=1) # OUT (256, 12, 12)
		# MAXPOOL((256, 12, 12)) = (256, 6, 6)

		self.decoder1 = nn.ConvTranspose2d(256, 128, 2, stride=2) # OUT (128, 12, 12)
		# dec1_cat((decoder1, bottleneck_conv), dim=1) -> OUT (384, 12, 12)
		self.dec_conv1 = nn.Conv2d(384, 128, 3, padding=1) # OUT(128, 12, 12)

		self.decoder2 = nn.ConvTranspose2d(128, 64, 2, stride=2) # OUT (384, 24, 24)
		# dec2_cat((decoder2, up_block2), dim=1) -> OUT (192, 24, 24)
		self.dec_conv2 = nn.Conv2d(192, 64, 3, padding=1) # OUT(64, 24, 24)

		self.decoder3 = nn.ConvTranspose2d(64, 32, 2, stride=2) # OUT (32, 48, 48)
		# dec3_cat((decoder3, up_block1), dim=1) -> OUT (96, 48, 48)
		self.dec_conv3 = nn.Conv2d(96, 64, 3, padding=1) # OUT(16, 48, 48)

		self.final_conv = nn.Conv2d(64, 1, 1)

	def forward(self, x):
		up1_no_pool = F.relu(self.up_block1(x))
		up1_pool = self.pool(up1_no_pool)

		up2_no_pool = F.relu(self.up_block2(up1_pool))
		up2_pool = self.pool(up2_no_pool)

		bottleneck_conv = F.relu(self.bottleneck_conv(up2_pool))
		bottleneck_conv_pool = self.pool(bottleneck_conv)

		dec1 = self.decoder1(bottleneck_conv_pool)
		dec1_cat = torch.cat((dec1, bottleneck_conv), dim=1)
		dec_conv1 = F.relu(self.dec_conv1(dec1_cat)) 

		dec2 = self.decoder2(dec_conv1)
		dec2_cat = torch.cat((dec2, up2_no_pool), dim=1)
		dec_conv2 = F.relu(self.dec_conv2(dec2_cat))

		dec3 = self.decoder3(dec_conv2)
		dec3_cat = torch.cat((dec3, up1_no_pool), dim=1)
		dec_conv3 = F.relu(self.dec_conv3(dec3_cat))

		output_conv = self.final_conv(dec_conv3)

		return output_conv
			

	

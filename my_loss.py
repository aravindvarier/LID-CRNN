import torch.nn as nn
import torch

class focalLoss:
	def __init__(self, weight, gamma, device, num_classes):
		self.weight = weight
		self.gamma = gamma
		self.device = device
		self.num_classes = num_classes

	def __call__(self, pred, labels):
		#step 1 - softmax
		prob = nn.Softmax(dim=1)(pred)
		#step 2 - onehot encoding the labels
		labels = labels.unsqueeze(1).cpu() #have to bring to cpu to scatter
		labels_onehot = torch.zeros(labels.shape[0], self.num_classes)
		labels_onehot.scatter_(1, labels, 1)
		labels_onehot = labels_onehot.to(self.device)
		#step 3 - calculate additional focal loss modulation factor
		modulating_factor = (1 - prob)**self.gamma
		#focal_weight = focal_weight.to(device)
		#step 4 - multiply them all together to get loss and add it all up
		L = (-1 * self.weight * modulating_factor * labels_onehot * torch.log(prob)).sum()
		return L


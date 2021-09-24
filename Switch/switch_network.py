import torch
import torch.nn as nn
import torchvision

def make_layers_vgg(cfg, in_ch=3, use_batch_norm=False):
    """
    Code borrowed from torchvision/models/vgg.py
    """
    layers = []

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_ch, v, kernel_size=3, padding=1)
            if use_batch_norm:
                layers.extend(
                    [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_ch = v

    return nn.Sequential(*layers)
	
# Switch classifier
class SwitchNet(nn.Module):
    
    def __init__(
            self,
            load_pretr_weights_vgg=False):
        super(SwitchNet, self).__init__()

        num_classes = 3
        # vgg16, corresponds to cfg['D'] from torchvision/models/vgg.py
        self.conv1_features = make_layers_vgg([64, 64, 'M'], in_ch=3)
        self.conv2_features = make_layers_vgg([128, 128, 'M'], in_ch=64)
        self.conv3_features = make_layers_vgg([256, 256, 256, 'M'], in_ch=128)
        self.conv4_features = make_layers_vgg([512, 512, 512, 'M'], in_ch=256)
        self.conv5_features = make_layers_vgg([512, 512, 512, 'M'], in_ch=512)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes))
			
        self.classify = nn.Sequential(nn.ReLU(),
                        nn.Softmax(dim=1))

        
        self._initialize_weights()

        if load_pretr_weights_vgg:
            pretr_dict = torchvision.models.vgg16(pretrained=True).state_dict()
            this_net_dict = self.state_dict()
            this_net_keys = list(this_net_dict.keys())

            for i, (pretr_key, pretr_tensor_val) in enumerate(pretr_dict.items()):
                # pretrained vgg16 keys start with 'features' or with 'classifier'
                if 'features' in pretr_key:
                    this_net_tensor_val = this_net_dict[this_net_keys[i]]
                    assert this_net_tensor_val.shape == pretr_tensor_val.shape
                    this_net_tensor_val.data = pretr_tensor_val.data.clone()
                    #print(pretr_key, pretr_tensor_val.shape)
                else:
                    break

            self.load_state_dict(this_net_dict)

    def forward(self, x):
        x = self.conv1_features(x)
        x = self.conv2_features(x)
        x = self.conv3_features(x)
        x = self.conv4_features(x)
        x = self.conv5_features(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) 
        x = self.fcs(x)
        x = self.classify(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

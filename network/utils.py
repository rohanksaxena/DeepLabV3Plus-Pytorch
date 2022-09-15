import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from collections import OrderedDict


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():  # 2 modules, ll features and hl features
            if self.hrnet_flag and name.startswith('transition'):  # if using hrnet, you need to take care of transition
                if name == 'transition1':  # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else:  # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else:  # other models (ex:resnet,mobilenet) are convolutions in series.

                x = module(x)
                m = nn.ReLU()
                output = m(x)
                feature_maps = output.squeeze(0)
                feature_maps = feature_maps.cpu().numpy()
                # mask = Image.open(
                #     'D:\\Workspaces\\Thesis\\deeplab\\DeepLabV3Plus-Pytorch\\Sooty_Albatross_0031_1066.png')  # Read image mask
                # mask = mask.resize((feature_maps.shape[2],
                #                     feature_maps.shape[1]))  # Reshape mask to spatial resolution of feature maps
                # mask = np.asarray(mask)
                # mask[mask > 0] = 255

                fig = plt.figure(figsize=(feature_maps.shape[1], feature_maps.shape[2]))
                if name == 'low_level_features':
                    ncols = 6
                    nrows = 4
                elif name == 'high_level_features':
                    ncols = 20
                    nrows = 16

                for i, map in enumerate(feature_maps):
                    # map = np.multiply(map, mask)
                    z = map.reshape(-1)
                    z = np.float32(z)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    K = 5
                    ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                    label = label.reshape((feature_maps.shape[1], feature_maps.shape[2]))
                    # label = np.multiply(label, mask)
                    fig.add_subplot(nrows, ncols, i + 1)
                    plt.imshow(label)
                    plt.axis('off')
                plt.suptitle(name)
                plt.savefig(f'{name}.jpg')
                plt.clf()
                plt.close()


                # Cluster pixels in channel dimension
                # Vectorize features
                maps = feature_maps.reshape((feature_maps.shape[0]), -1).T
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                K = 5
                attempts = 10
                ret, label, center = cv2.kmeans(maps, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
                result_image = label.reshape((feature_maps.shape[1], feature_maps.shape[2]))
                plt.imshow(result_image)
                plt.axis('off')
                plt.savefig(f'{name}_clustered.jpg')
                plt.clf()
                plt.close()



            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag:  # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out


import torch.nn as nn
import torch.nn.functional as F
import torch

class CPM_hand_torch(nn.Module):
    def __init__(self, k):
        super(CPM_hand_torch, self).__init__()
        self.k = k
        #self.initialize_weights()
        # 设输入为x*x
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)  # (x + 2 - 9) / 8 + 1,若输入为368，则输出为45 * 45
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4) 
        self.conv1_bn_stage1 = nn.BatchNorm2d(128)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv2_bn_stage1 = nn.BatchNorm2d(128)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_bn_stage1 = nn.BatchNorm2d(128)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv4_bn_stage1 = nn.BatchNorm2d(32)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv5_bn_stage1 = nn.BatchNorm2d(512)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv6_bn_stage1 = nn.BatchNorm2d(512)
        self.conv7_stage1 = nn.Conv2d(512, self.k + 1, kernel_size=1) # 关节数+1


        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv1_bn_stage2 = nn.BatchNorm2d(128)
        self.pool1_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv2_bn_stage2 = nn.BatchNorm2d(128)
        self.pool2_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_bn_stage2 = nn.BatchNorm2d(128)
        self.pool3_stage2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv4_bn_stage2 = nn.BatchNorm2d(32)

        self.Mconv1_stage2 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5) # 32为特征图的通道数，2表示每一类别的置信度和位置？
        self.Mconv1_bn_stage2 = nn.BatchNorm2d(128)
        self.Mconv2_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv2_bn_stage2 = nn.BatchNorm2d(128)
        self.Mconv3_stage2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_bn_stage2 = nn.BatchNorm2d(128)
        self.Mconv4_stage2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv4_bn_stage2 = nn.BatchNorm2d(128)
        self.Mconv5_stage2 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)# 每一stage的输出都为k+1维

        self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_bn_stage3 = nn.BatchNorm2d(32)

        self.Mconv1_stage3 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv1_bn_stage3 = nn.BatchNorm2d(128)
        self.Mconv2_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv2_bn_stage3 = nn.BatchNorm2d(128)
        self.Mconv3_stage3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_bn_stage3 = nn.BatchNorm2d(128)
        self.Mconv4_stage3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv4_bn_stage3 = nn.BatchNorm2d(128)
        self.Mconv5_stage3 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0) # 和stage2一样，对belief map与特征图进行融合后，进行卷积提取特征

        self.conv1_stage4 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_bn_stage4 = nn.BatchNorm2d(32)

        self.Mconv1_stage4 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv1_bn_stage4 = nn.BatchNorm2d(128)
        self.Mconv2_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv2_bn_stage4 = nn.BatchNorm2d(128)
        self.Mconv3_stage4 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_bn_stage4 = nn.BatchNorm2d(128)
        self.Mconv4_stage4 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv4_bn_stage4 = nn.BatchNorm2d(128)
        self.Mconv5_stage4 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage5 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_bn_stage5 = nn.BatchNorm2d(32)

        self.Mconv1_stage5 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv1_bn_stage5 = nn.BatchNorm2d(128)
        self.Mconv2_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv2_bn_stage5 = nn.BatchNorm2d(128)
        self.Mconv3_stage5 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_bn_stage5 = nn.BatchNorm2d(128)
        self.Mconv4_stage5 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv4_bn_stage5 = nn.BatchNorm2d(128)
        self.Mconv5_stage5 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)

        self.conv1_stage6 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv1_bn_stage6 = nn.BatchNorm2d(32)

        self.Mconv1_stage6 = nn.Conv2d(32 + self.k + 2, 128, kernel_size=11, padding=5)
        self.Mconv1_bn_stage6 = nn.BatchNorm2d(128)
        self.Mconv2_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv2_bn_stage6 = nn.BatchNorm2d(128)
        self.Mconv3_stage6 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_bn_stage6 = nn.BatchNorm2d(128)
        self.Mconv4_stage6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv4_bn_stage6 = nn.BatchNorm2d(128)
        self.Mconv5_stage6 = nn.Conv2d(128, self.k + 1, kernel_size=1, padding=0)



    def _stage1(self, image):
        # stage1：输入原始图，经过conv与pool，输出(k+1) * w' * h'的热力图
        x = self.pool1_stage1(F.relu(self.conv1_bn_stage1(self.conv1_stage1(image))))
        x = self.pool2_stage1(F.relu(self.conv2_bn_stage1(self.conv2_stage1(x))))
        x = self.pool3_stage1(F.relu(self.conv3_bn_stage1(self.conv3_stage1(x))))
        x = F.relu(self.conv4_bn_stage1(self.conv4_stage1(x)))
        x = F.relu(self.conv5_bn_stage1(self.conv5_stage1(x)))
        x = F.relu(self.conv6_bn_stage1(self.conv6_stage1(x)))
        x = self.conv7_stage1(x)
        
        return x

    def _middle(self, image):
        # 中间层，对原始图像提取特征图，准备与上一stage的热力图融合
        x = self.pool1_stage2(F.relu(self.conv1_bn_stage2(self.conv1_stage2(image))))
        x = self.pool2_stage2(F.relu(self.conv2_bn_stage2(self.conv2_stage2(x))))
        x = self.pool3_stage2(F.relu(self.conv3_bn_stage2(self.conv3_stage2(x))))

        return x

    def _stage2(self, pool3_stage2_map, conv7_stage1_map, pool_center_map):
    
        x = F.relu(self.conv4_bn_stage2(self.conv4_stage2(pool3_stage2_map)))
        x = torch.cat([x, conv7_stage1_map, pool_center_map], axis=1) # 数据融合就是简单的堆一起是吧？
        x = F.relu(self.Mconv1_bn_stage2(self.Mconv1_stage2(x)))
        x = F.relu(self.Mconv2_bn_stage2(self.Mconv2_stage2(x)))
        x = F.relu(self.Mconv3_bn_stage2(self.Mconv3_stage2(x)))
        x = F.relu(self.Mconv4_bn_stage2(self.Mconv4_stage2(x)))
        x = self.Mconv5_stage2(x)

        return x

    def _stage3(self, pool3_stage2_map, Mconv5_stage2_map, pool_center_map):

        x = F.relu(self.conv1_bn_stage3(self.conv1_stage3(pool3_stage2_map)))
        x = torch.cat([x, Mconv5_stage2_map, pool_center_map], axis=1)
        x = F.relu(self.Mconv1_bn_stage3(self.Mconv1_stage3(x)))
        x = F.relu(self.Mconv2_bn_stage3(self.Mconv2_stage3(x)))
        x = F.relu(self.Mconv3_bn_stage3(self.Mconv3_stage3(x)))
        x = F.relu(self.Mconv4_bn_stage3(self.Mconv4_stage3(x)))
        x = self.Mconv5_stage3(x)

        return x

    def _stage4(self, pool3_stage2_map, Mconv5_stage3_map, pool_center_map):

        x = F.relu(self.conv1_bn_stage4(self.conv1_stage4(pool3_stage2_map)))
        x = torch.cat([x, Mconv5_stage3_map, pool_center_map], axis=1)
        x = F.relu(self.Mconv1_bn_stage4(self.Mconv1_stage4(x)))
        x = F.relu(self.Mconv2_bn_stage4(self.Mconv2_stage4(x)))
        x = F.relu(self.Mconv3_bn_stage4(self.Mconv3_stage4(x)))
        x = F.relu(self.Mconv4_bn_stage4(self.Mconv4_stage4(x)))
        x = self.Mconv5_stage4(x)

        return x

    def _stage5(self, pool3_stage2_map, Mconv5_stage4_map, pool_center_map):

        x = F.relu(self.conv1_bn_stage5(self.conv1_stage5(pool3_stage2_map)))
        x = torch.cat([x, Mconv5_stage4_map, pool_center_map], axis=1)
        x = F.relu(self.Mconv1_bn_stage5(self.Mconv1_stage5(x)))
        x = F.relu(self.Mconv2_bn_stage5(self.Mconv2_stage5(x)))
        x = F.relu(self.Mconv3_bn_stage5(self.Mconv3_stage5(x)))
        x = F.relu(self.Mconv4_bn_stage5(self.Mconv4_stage5(x)))
        x = self.Mconv5_stage5(x)

        return x

    def _stage6(self, pool3_stage2_map, Mconv5_stage5_map, pool_center_map):
        
        x = F.relu(self.conv1_bn_stage6(self.conv1_stage6(pool3_stage2_map)))
        x = torch.cat([x, Mconv5_stage5_map, pool_center_map], axis=1)
        x = F.relu(self.Mconv1_bn_stage6(self.Mconv1_stage6(x)))
        x = F.relu(self.Mconv2_bn_stage6(self.Mconv2_stage6(x)))
        x = F.relu(self.Mconv3_bn_stage6(self.Mconv3_stage6(x)))
        x = F.relu(self.Mconv4_bn_stage6(self.Mconv4_stage6(x)))
        x = self.Mconv5_stage6(x)
        # lxd add
        # x = F.sigmoid(x)
        return x


    def forward(self, image, center_map):

        # 1. center_map为是表示人体中心点位置gaussion分布的feature map
        pool_center_map = self.pool_center(center_map)
        # 2. 原始图像输入stage1，获得stage1的热力图
        conv7_stage1_map = self._stage1(image)
        # 3. middle用于产生在stage2~6中使用的特征图，输入为原始图像image
        pool3_stage2_map = self._middle(image)
        # 4. 在stage2~6中，输入上一stage的heat map、中间层特征图与center_map，cat在一起后进行卷积提取特征，输入维度为(k + 1) + 3 + 输出(k+1)维的热力图
        Mconv5_stage2_map = self._stage2(pool3_stage2_map, conv7_stage1_map,
                                         pool_center_map)
        Mconv5_stage3_map = self._stage3(pool3_stage2_map, Mconv5_stage2_map,
                                         pool_center_map)
        Mconv5_stage4_map = self._stage4(pool3_stage2_map, Mconv5_stage3_map,
                                         pool_center_map)
        Mconv5_stage5_map = self._stage5(pool3_stage2_map, Mconv5_stage4_map,
                                         pool_center_map)
        Mconv5_stage6_map = self._stage6(pool3_stage2_map, Mconv5_stage5_map,
                                         pool_center_map)

        return conv7_stage1_map, Mconv5_stage2_map, Mconv5_stage3_map, Mconv5_stage4_map, Mconv5_stage5_map, Mconv5_stage6_map
import torch.nn as nn
from abc import ABC, abstractmethod


class PhysionFeatureExtractor(nn.Module, ABC):

    def __init__(self,):
        '''
        weights_path: path to the weights of the model
        '''
        super().__init__()

    @abstractmethod
    def transform(self):
        '''
        :return: Image Transform, Frame Gap, Minimum Number of Frames
        '''
        pass

    @abstractmethod
    def extract_features(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''
        pass

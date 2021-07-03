import torch

from constants import *
from utils import *
from game import *
from nn import *



def move_translation_test():
    for i in range(1, 12):
        move = move_from_int(i)
        coord = coord_from_int(i)

        assert int_from_coord(coord) == i
        assert int_from_move(move) == i

    print("translation test passed")


def nn_test():
    test_input = torch.zeros((10,3,14,14))

    block = torch.nn.Conv2d(3, 7, kernel_size=3, padding=(1,1))

    test_output = block(test_input)
    assert tuple(test_output.shape) == (10, 7, 14, 14)

    resblock = Res2d(7, kernel_size=5)

    test_output_2 = resblock(test_output)
    assert tuple(test_output_2.shape) == (10, 7, 14, 14)

    resnet = ResNet(7, kernel_size=3, depth=3)

    test_output_3 = resblock(test_output_2)
    assert tuple(test_output_3.shape) == (10, 7, 14, 14)

    
    a = torch.zeros((20, 14, 5, 5))

    head = ActionHead(in_channels=14, phase="placing")

    a0 = head(a)

    assert tuple(a0.shape) == (20, 25)

    print("nn test passed")



move_translation_test()
nn_test()

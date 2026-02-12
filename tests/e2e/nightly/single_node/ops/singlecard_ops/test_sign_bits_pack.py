import torch
import torch_npu
import torchair
import torch.nn as nn
import torch._dynamo
from torch_npu.testing.testcase import TestCase, run_tests

from vllm_ascend.utils import enable_custom_op
enable_custom_op()

torch._logging.set_logs(graph_code=True)

class TestTorchCompileSignBitsPack(TestCase):

    def test_custom_sign_bits_pack_egaer(self):
        print(f'test_custom_sign_bits_pack_egaer begin')
        self_ = torch.tensor([5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2,5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2,2],dtype=torch.float32).npu()
        print(f'self_ = {self_.shape}')
        # torch_npu调用方式
        r2 = torch_npu.npu_sign_bits_pack(self_, 2)
        print(f'r2 = {r2}')

        r3 = torch_npu.npu_sign_bits_pack(self_, 1)
        print(f'r3 = {r3}')

        
        cr2 = torch.ops._C_ascend.npu_sign_bits_pack(self_, 2)
        print(f'cr2 = {cr2}')

        cr3 = torch.ops._C_ascend.npu_sign_bits_pack(self_, 1)
        print(f'cr3 = {cr3}')
        print(f'test_custom_sign_bits_pack_egaer end')


    def test_custom_sign_bits_pack_graph(self):
        DEVICE_ID = 0
        # start run custom ops
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, input, size):
                return torch.ops._C_ascend.npu_sign_bits_pack(input, size)

        print(f'test_custom_sign_bits_pack_graph begin')

        input_ = torch.tensor([5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2,5,4,3,2,0,-1,-2, 4,3,2,1,0,-1,-2,2],dtype=torch.float32)

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        torch.npu.set_device(DEVICE_ID)
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_out = npu_mode(input_.to('npu:0'), 2)
        print(f'sign_bits_pack = {npu_out}')

        print(f'test_custom_sign_bits_pack_graph end')



if __name__ == "__main__":
    run_tests()

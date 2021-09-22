import io
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from openvino.inference_engine import IECore

from .. import dataloaders
from .kpconv import KPFCNN


class KPFCNN_OV(nn.Module):

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, inputs):
        tensors = namedtuple('KPConvBatch', [])
        tensors.features = inputs['features']
        tensors.points = inputs['points']
        tensors.neighbors = inputs['neighbors']
        tensors.pools = inputs['pools']
        tensors.upsamples = inputs['upsamples']
        return self.base_model(tensors)


class OpenVINOModel:

    def __init__(self, base_model):
        self.ie = IECore()
        self.exec_net = None

        if isinstance(base_model, KPFCNN):
            self.base_model = KPFCNN_OV(base_model)
        else:
            self.base_model = base_model

        # A workaround for unsupported torch.square by ONNX
        torch.square = lambda x: torch.pow(x, 2)

    def _get_input_names(self, inputs):
        names = []
        for name, tensor in inputs.items():
            if isinstance(tensor, list):
                for i in range(len(tensor)):
                    names.append(name + str(i))
            else:
                names.append(name)
        return names

    def _get_inputs(self, inputs, export=False):
        if isinstance(inputs, dataloaders.concat_batcher.KPConvBatch):
            inputs = {
                'features': inputs.features,
                'points': inputs.points,
                'neighbors': inputs.neighbors,
                'pools': inputs.pools,
                'upsamples': inputs.upsamples,
            }
        return inputs

    def _read_torch_model(self, inputs):
        input_names = self._get_input_names(inputs)

        buf = io.BytesIO()
        torch.onnx.export(self.base_model, inputs, buf, input_names=input_names)
        # torch.onnx.export(self.base_model, inputs, 'kpconv.onnx', input_names=input_names)
        net = self.ie.read_network(buf.getvalue(), b'', init_from_buffer=True)
        self.exec_net = self.ie.load_network(net, 'CPU')

    def forward(self, inputs):
        inputs = self._get_inputs(inputs)
        if self.exec_net is None:
            self._read_torch_model(inputs)

        tensors = {}
        for name, tensor in inputs.items():
            if name == 'labels':
                continue
            if isinstance(tensor, list):
                for i in range(len(tensor)):
                    if tensor[i].nelement() > 0:
                        tensors[name + str(i)] = tensor[i].detach().numpy()
            else:
                if tensor.nelement() > 0:
                    tensors[name] = tensor.detach().numpy()

        output = self.exec_net.infer(tensors)
        output = next(iter(output.values()))
        return torch.tensor(output)

    def __call__(self, inputs):
        return self.forward(inputs)

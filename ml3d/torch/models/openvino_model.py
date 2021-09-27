import io
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from openvino.inference_engine import IECore

from .. import dataloaders


class OpenVINOModel:

    def __init__(self, base_model):
        self.ie = IECore()
        self.exec_net = None
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
        tensors = self._get_inputs(inputs)
        input_names = self._get_input_names(tensors)

        # Forward origin inputs instead of export <tensors>
        origin_forward = self.base_model.forward
        self.base_model.forward = lambda x: origin_forward(inputs)

        buf = io.BytesIO()
        torch.onnx.export(self.base_model, tensors, buf, input_names=input_names)
        # torch.onnx.export(self.export_model, inputs, 'kpconv.onnx', input_names=input_names, opset_version=11)
        self.base_model.forward = origin_forward

        net = self.ie.read_network(buf.getvalue(), b'', init_from_buffer=True)
        self.exec_net = self.ie.load_network(net, 'CPU')

    def forward(self, inputs):
        if self.exec_net is None:
            self._read_torch_model(inputs)

        inputs = self._get_inputs(inputs)

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

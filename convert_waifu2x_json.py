import json

import torch

from waifu2x_upcunet import UpCunet


def convert(w2x_json_fname):
    if w2x_json_fname.endswith('.json'):
        name = w2x_json_fname[:-5]
    else:
        name = w2x_json_fname

    model = UpCunet(channels=3)
    with open(w2x_json_fname, 'r') as in_f:
        w2x_json = json.load(in_f)

    params = list(model.parameters())

    # Relies on the ordering of the weights to magically just be the same
    # In practice, this actually seems to work
    # (Though it's certainly not guaranteed to)
    for tensor_weight, tensor_bias, w2x_data in zip(params[::2], params[1::2], w2x_json):
        weight = torch.Tensor(w2x_data['weight'])
        tensor_weight.data = weight

        bias = torch.Tensor(w2x_data['bias'])
        tensor_bias.data = bias
        
    torch.save(model, f'{name}.pt')
    
    trace_input = torch.rand([1, 3, 256, 256])
    dynamic_axes = {'image': {0: 'batch', 2: 'height', 3: 'width'}}
    torch.onnx.export(
        model=model, 
        args=(trace_input,),
        f=f'{name}.onnx',
        input_names=('image',),
        output_names=('upscale',),
        dynamic_axes=dynamic_axes,
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) <= 1:
        print('Usage: convert_waifu2x_json.py <model json file>')
        exit()
        
    convert(sys.argv[1])

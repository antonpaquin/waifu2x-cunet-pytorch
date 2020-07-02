# Waifu2x UpCunet on Pytorch

The [waifu2x website](http://waifu2x.udp.jp/) uses (as far as I can tell) the "cunet/art" class of models which can be found in [Nagadomi's repo](https://github.com/nagadomi/waifu2x/tree/master/models/cunet/art).

Those official models are written in the old lua torch, which inconvenient as support for running that kind of model is going away.

After searching (admittedly only a little), I wasn't able to find a clean pytorch converter or reimplementation. This repo is my attempt to remedy this situation.

- `waifu2x_upcunet.py` 

is a pytorch clone of the original model architecture described [here](https://github.com/nagadomi/waifu2x/blob/master/lib/srcnn.lua) (line 575).

- `convert_waifu2x_json.py` 

is a utility script that converts one of the weight files [here](https://github.com/nagadomi/waifu2x/tree/master/models/cunet/art) into a pytorch saved model, and an exported onnx model for good measure.

Use it like:

```
python convert_waifu2x_json.py noise1_scale2.0x_model.json
```

This will generate `noise1_scale2.0x_model.pt` and `noise1_scale2.0x_model.onnx` alongside the json model.

Requirements: torch, Pillow, numpy

# Exported Objects

Here are some exported versions of the 2xScale 1xNoiseReduction model:

[onnx](https://hivemind-repo.s3-us-west-2.amazonaws.com/obj/noise1_scale2.0x_model.onnx)

[pytorch](https://hivemind-repo.s3-us-west-2.amazonaws.com/obj/noise1_scale2.0x_model.pt)

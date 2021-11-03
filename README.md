# ConvTranspose4d-PyTorch

Implementation of ConvTranpose4d for PyTorch

This code was inspired from [this project](https://github.com/ZhengyuLiang24/Conv4d-PyTorch).

You can read their [README](https://github.com/matheusja/ConvTranspose4d-PyTorch/blob/main/README_old.md) for some context.

I tried getting ```padding```, ```output_padding``` and ```stride``` to work like they work on pytorch analogues for lower dimensions, more espifically [ConvTranspose1d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html), [ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) and [ConvTranspose3d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html)

Some features were not implemented.

I used the same strategy to validate my design as the people I forked - create a ConvTranspose2d with the same style for implementation and then compare the results with the implementation available in the pytorch library.

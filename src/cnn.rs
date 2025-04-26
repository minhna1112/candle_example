use candle_core::{Device, Result, Tensor};
use candle_nn::{BatchNorm2d, Conv2d, Linear, Module};
struct Cnn {
    first: Conv2d,
    bn: BatchNorm2d,
    second: Conv2d,
    third: Linear,
}

impl Model for Cnn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(x)?;
        let x = self.bn.forward(x)?;
        let x = self.second.forward(x)?;
        let x = x.reshape([x.dims()[0], x.dims()[1] * x.dims()[2] * x.dims()[3]])?;
        self.third.forward(x)
    }
}

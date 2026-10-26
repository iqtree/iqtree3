use std::ops::Deref;

use candle_core::{CpuStorage, CustomOp1, Storage, Tensor};


pub fn log_gamma_pdf(x: &Tensor, alpha: &Tensor) -> Tensor {
    let lgamma_alpha = alpha.apply_op1(LgammaOp).unwrap();

    let lgamma_alpha_b = lgamma_alpha.broadcast_as(x.shape()).unwrap();
    let alpha_b = alpha.broadcast_as(x.shape()).unwrap();

    let log_pdf = ((&alpha_b * &alpha_b.log().unwrap() - lgamma_alpha_b).unwrap() + (&alpha_b - 1.0) * x.log().unwrap()).unwrap() - &alpha_b * x;
    log_pdf.unwrap()
}

/// Custom op that computes lgamma(x) elementwise.
/// Backward uses the digamma function (derivative of lgamma).
#[derive(Debug, Clone)]
pub struct LgammaOp;

impl CustomOp1 for LgammaOp {
    fn name(&self) -> &'static str {
        "LgammaOp"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(CpuStorage, candle_core::Shape)> {
        let data = match storage {
            CpuStorage::F64(data) => data,
            _ => panic!("LgammaOp: expected f64 storage"),
        };

        let output = candle_core::cpu_backend::unary_map(data, layout, spec_math::cephes64::lgam);
        Ok((CpuStorage::F64(output), layout.shape().clone()))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &candle_core::Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (storage, layout) = arg.storage_and_layout();

        let data = match storage.deref() {
            Storage::Cpu(CpuStorage::F64(data)) => data,
            _ => panic!("LgammaOp: expected cpu f64 storage in bwd"),
        };

        let grads = candle_core::cpu_backend::unary_map(data, &layout, spec_math::cephes64::psi);
        let grad_tensor = Tensor::from_vec(grads, arg.shape(), &candle_core::Device::Cpu)?;

        Ok(Some(grad_tensor.mul(grad_res)?))
    }
}

#[derive(Debug, Clone)]
pub struct GammaOp {
    categories: usize,
}

impl GammaOp {
    pub fn new(categories: usize) -> Self {
        Self { categories }
    }
}

impl candle_core::CustomOp1 for GammaOp {
    fn name(&self) -> &'static str {
        "GammaOp"
    }
    
    fn cpu_fwd(&self, storage: &candle_core::CpuStorage, layout: &candle_core::Layout) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let alpha = match storage {
            candle_core::CpuStorage::F64(v) => v[layout.start_offset()],
            _ => panic!("Expected f64 storage for GammaOp"),
        };

        let bounds = (0..=self.categories).map(|k| {
            //gamma.inverse_cdf(k as f64 / self.categories as f64)
            spec_math::cephes64::gdtri(alpha, alpha, k as f64 / self.categories as f64)
        }).collect::<Vec<f64>>();

        let categories = bounds.iter().zip(bounds.iter().skip(1)).map(|(a, b)| {
            //let mean = gamma_alpha1.cdf(*b) - gamma_alpha1.cdf(*a);
            // a and b seems to be swapped here.
            let mean = spec_math::cephes64::gdtr(alpha, alpha + 1.0, *b) - spec_math::cephes64::gdtr(alpha, alpha + 1.0, *a);
            mean * self.categories as f64
        }).collect::<Vec<f64>>();

        let output = CpuStorage::F64(categories);
        let shape = candle_core::Shape::from(&[self.categories]);

        Ok((output, shape))
    }

    fn bwd(&self, _arg: &candle_core::Tensor, _res: &candle_core::Tensor, _grad_res: &candle_core::Tensor) -> candle_core::Result<Option<candle_core::Tensor>> {
        
        let h = 1e-5;

        let h_plus = _arg + candle_core::Tensor::full(h, &[], _arg.device())?;
        let h_minus = _arg - candle_core::Tensor::full(h, &[], _arg.device())?;

        let derivative = (h_plus?.apply_op1(self.clone())? - h_minus?.apply_op1(self.clone())?)? / (2.0 * h);

        let grad = (_grad_res * derivative?)?.sum_all()?;
        Ok(Some(grad))
    }
}


#[cfg(test)]
mod test {
    use candle_core::{Tensor, Var};

    use super::*;
    #[test]
    fn test_gamma_op() {
        let op = GammaOp::new(4);
        let alpha = 0.08;
        
        let alpha = Tensor::from_vec(vec![alpha], &[], &candle_core::Device::Cpu).unwrap();
        let alpha = Var::from_tensor(&alpha).unwrap();

        let categories = alpha.apply_op1(op).unwrap();

        println!("{:?}", categories);

        

        println!("{:?}", categories.get(0).unwrap().backward().unwrap());
    }
}
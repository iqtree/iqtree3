use std::sync::{Arc, Mutex};

use candle_core::CpuStorage;

pub struct FelsensteinOp {
    felsenstein : Arc<Mutex<phylo_grad::FelsensteinTree<20>>>,
    result_storage: Mutex<Option<phylo_grad::FelsensteinResult<20>>>,
}
pub struct FelsensteinWithEdgeOp {
    felsenstein : Arc<Mutex<phylo_grad::FelsensteinTree<20>>>,
    result_storage: Mutex<Option<phylo_grad::FelsensteinResultWithTree<20>>>,
}

#[derive(Clone, Debug)]
pub struct FelsensteinWithEdgeFwdOp {
    felsenstein : Arc<Mutex<phylo_grad::FelsensteinTree<20>>>,
}

#[derive(Clone, Debug)]
pub struct FelsensteinFwdOp {
    felsenstein: Arc<Mutex<phylo_grad::FelsensteinTree<20>>>,
}

impl Clone for FelsensteinOp {
    fn clone(&self) -> Self {
        Self {
            felsenstein: Arc::clone(&self.felsenstein),
            result_storage: Mutex::new(None),
        }
    }
}

impl Clone for FelsensteinWithEdgeOp {
    fn clone(&self) -> Self {
        Self {
            felsenstein: Arc::clone(&self.felsenstein),
            result_storage: Mutex::new(None),
        }
    }
}

fn tensor2nalgebra2d(s: &candle_core::CpuStorage, l: &candle_core::Layout) -> Vec<phylo_grad::nalgebra::SMatrix<f64, 20, 20>> {
    let dims = l.shape().dims();
    let stride = l.stride();
    assert!(dims.len() == 3);
    let batch_size = dims[0];
    let mut result = Vec::with_capacity(batch_size);
    match s {
        CpuStorage::F64(data) => {
            let mut offset = l.start_offset();
            for _ in 0..batch_size {
                let mut mat = phylo_grad::nalgebra::SMatrix::<f64, 20, 20>::zeros();
                for i in 0..20 {
                    for j in 0..20 {
                        mat[(i, j)] = data[offset + i * stride[1] + j * stride[2]];
                    }
                }
                result.push(mat);
                offset += stride[0];
            }
        }
        _ => panic!("Expected f64 storage."),
    }
    result
}

fn tensor2nalgebra1d(s: &candle_core::CpuStorage, l: &candle_core::Layout) -> Vec<phylo_grad::nalgebra::SVector<f64, 20>> {
    let dims = l.shape().dims();
    let stride = l.stride();
    assert!(dims.len() == 2);
    let batch_size = dims[0];
    let mut result = Vec::with_capacity(batch_size);
    match s {
        CpuStorage::F64(data) => {
            let mut offset = l.start_offset();
            for _ in 0..batch_size {
                let mut vec = phylo_grad::nalgebra::SVector::<f64, 20>::zeros();
                for i in 0..20 {
                    vec[i] = data[offset + i * stride[1]];
                }
                result.push(vec);
                offset += stride[0];
            }
        }
        _ => panic!("Expected f64 storage."),
    }
    result
}

fn tensor2vec(s: &candle_core::CpuStorage, l: &candle_core::Layout) -> Vec<f64> {
    let dims = l.shape().dims();
    let stride = l.stride();
    assert!(dims.len() == 1);
    let batch_size = dims[0];
    let mut result = Vec::with_capacity(batch_size);
    match s {
        CpuStorage::F64(data) => {
            let mut offset = l.start_offset();
            for _ in 0..batch_size {
                result.push(data[offset]);
                offset += stride[0];
            }
        }
        _ => panic!("Expected f64 storage."),
    }
    result
}

fn nalgebra2tensor1d(vec: &Vec<phylo_grad::nalgebra::SVector<f64, 20>>) -> candle_core::Tensor {
    let batch_size = vec.len();
    let mut data = vec![0f64; batch_size * 20];
    for (b, v) in vec.iter().enumerate() {
        for i in 0..20 {
            data[b * 20 + i] = v[i];
        }
    }
    candle_core::Tensor::from_vec(data, &candle_core::Shape::from(&[batch_size, 20]), &candle_core::Device::Cpu).unwrap()
}

fn nalgebra2tensor2d(mats: &Vec<phylo_grad::nalgebra::SMatrix<f64, 20, 20>>) -> candle_core::Tensor {
    let batch_size = mats.len();
    let mut data = vec![0f64; batch_size * 20 * 20];
    for (b, m) in mats.iter().enumerate() {
        for i in 0..20 {
            for j in 0..20 {
                data[b * 20 * 20 + i * 20 + j] = m[(i, j)];
            }
        }
    }
    candle_core::Tensor::from_vec(data, &candle_core::Shape::from(&[batch_size, 20, 20]), &candle_core::Device::Cpu).unwrap()
}

impl FelsensteinFwdOp {
    pub fn new(felsenstein: Arc<Mutex<phylo_grad::FelsensteinTree<20>>>) -> Self {
        Self {
            felsenstein,
        }
    }
}

impl FelsensteinWithEdgeOp {
    pub fn new(felsenstein: Arc<Mutex<phylo_grad::FelsensteinTree<20>>>) -> Self {
        Self {
            felsenstein,
            result_storage: Mutex::new(None),
        }
    }
    pub fn into_fwd_op(&self) -> FelsensteinWithEdgeFwdOp {
        FelsensteinWithEdgeFwdOp {
            felsenstein: Arc::clone(&self.felsenstein),
        }
    }
}

impl candle_core::CustomOp2 for FelsensteinFwdOp {

    fn name(&self) -> &'static str {
        "FelsensteinFwdOp"
    }

    /// S, sqrt
    fn cpu_fwd(
            &self,
            s1: &candle_core::CpuStorage,
            l1: &candle_core::Layout,
            s2: &candle_core::CpuStorage,
            l2: &candle_core::Layout,
        ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        
        let s = tensor2nalgebra2d(s1, l1);
        let sqrt_pi = tensor2nalgebra1d(s2, l2);

        let mut felsenstein = self.felsenstein.lock().unwrap();
        let result = felsenstein.calculate_likelihoods(&s, &sqrt_pi);

        let shape = candle_core::Shape::from(&[result.len()]);
        let cpu_storage = candle_core::CpuStorage::F64(result);
        return Ok((cpu_storage, shape))
    }
}

impl FelsensteinOp {
    pub fn new(felsenstein: Arc<Mutex<phylo_grad::FelsensteinTree<20>>>) -> Self {
        Self {
            felsenstein,
            result_storage: Mutex::new(None),
        }
    }

    pub fn into_fwd_op(&self) -> FelsensteinFwdOp {
        FelsensteinFwdOp {
            felsenstein: Arc::clone(&self.felsenstein),
        }
    }

    pub fn into_with_edge_op(&self) -> FelsensteinWithEdgeOp {
        FelsensteinWithEdgeOp {
            felsenstein: Arc::clone(&self.felsenstein),
            result_storage: Mutex::new(None),
        }
    }

    pub fn into_with_edge_fwd_op(&self) -> FelsensteinWithEdgeFwdOp {
        FelsensteinWithEdgeFwdOp {
            felsenstein: Arc::clone(&self.felsenstein),
        }
    }
}

impl candle_core::CustomOp2 for FelsensteinOp {

    fn name(&self) -> &'static str {
        "FelsensteinOp"
    }

    /// S, sqrt
    fn cpu_fwd(
            &self,
            s1: &candle_core::CpuStorage,
            l1: &candle_core::Layout,
            s2: &candle_core::CpuStorage,
            l2: &candle_core::Layout,
        ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        
        let s = tensor2nalgebra2d(s1, l1);
        let sqrt_pi = tensor2nalgebra1d(s2, l2);

        let mut felsenstein = self.felsenstein.lock().unwrap();
        let mut result = felsenstein.calculate_gradients(&s, &sqrt_pi);

        let likelihood = std::mem::take(&mut result.log_likelihood);

        let mut storage = self.result_storage.lock().unwrap();
        *storage = Some(result);

        let shape = candle_core::Shape::from(&[likelihood.len()]);
        let cpu_storage = candle_core::CpuStorage::F64(likelihood);
        return Ok((cpu_storage, shape))
    }

    fn bwd(
            &self,
            _arg1: &candle_core::Tensor,
            _arg2: &candle_core::Tensor,
            _res: &candle_core::Tensor,
            _grad_res: &candle_core::Tensor,
        ) -> candle_core::Result<(Option<candle_core::Tensor>, Option<candle_core::Tensor>)> {
        let mut result = self.result_storage.lock().unwrap();
        

        let grad_s = std::mem::take(&mut result.as_mut().unwrap().grad_s);
        let grad_s_tensor = nalgebra2tensor2d(&grad_s);
        let grad_sqrt_pi = std::mem::take(&mut result.as_mut().unwrap().grad_sqrt_pi);
        let grad_sqrt_pi_tensor = nalgebra2tensor1d(&grad_sqrt_pi);

        let grad_2 = _grad_res.unsqueeze(1).unwrap().unsqueeze(2).unwrap();
        let grad_1 = _grad_res.unsqueeze(1).unwrap();

        Ok((
            Some(grad_2.broadcast_mul(&grad_s_tensor).unwrap()),
            Some(grad_1.broadcast_mul(&grad_sqrt_pi_tensor).unwrap()),
        ))

    }
}


impl candle_core::CustomOp3 for FelsensteinWithEdgeOp {

    fn name(&self) -> &'static str {
        "FelsensteinWithEdgeOp"
    }

    /// S, sqrt
    fn cpu_fwd(
            &self,
            s1: &candle_core::CpuStorage,
            l1: &candle_core::Layout,
            s2: &candle_core::CpuStorage,
            l2: &candle_core::Layout,
            s3: &candle_core::CpuStorage,
            l3: &candle_core::Layout,
        ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        
        let s = tensor2nalgebra2d(s1, l1);
        let sqrt_pi = tensor2nalgebra1d(s2, l2);
        let branch_lengths = tensor2vec(s3, l3);

        let mut felsenstein = self.felsenstein.lock().unwrap();
        let mut result = felsenstein.calculate_gradients_with_branch_lengths(&s, &sqrt_pi, &branch_lengths);

        let likelihood = std::mem::take(&mut result.log_likelihood);

        let mut storage = self.result_storage.lock().unwrap();
        *storage = Some(result);

        let shape = candle_core::Shape::from(&[likelihood.len()]);
        let cpu_storage = candle_core::CpuStorage::F64(likelihood);
        return Ok((cpu_storage, shape))
    }

    fn bwd(
            &self,
            _arg1: &candle_core::Tensor,
            _arg2: &candle_core::Tensor,
            _arg3: &candle_core::Tensor,
            _res: &candle_core::Tensor,
            grad_res: &candle_core::Tensor,
        ) -> candle_core::Result<(Option<candle_core::Tensor>, Option<candle_core::Tensor>, Option<candle_core::Tensor>)> {
        let mut result = self.result_storage.lock().unwrap();
        

        let grad_s = std::mem::take(&mut result.as_mut().unwrap().grad_s);
        let grad_s_tensor = nalgebra2tensor2d(&grad_s);
        let grad_sqrt_pi = std::mem::take(&mut result.as_mut().unwrap().grad_sqrt_pi);
        let grad_sqrt_pi_tensor = nalgebra2tensor1d(&grad_sqrt_pi);
        let grad_branch_lengths = std::mem::take(&mut result.as_mut().unwrap().grad_tree);
        let L = grad_branch_lengths.len();
        let num_branch = grad_branch_lengths[0].len();
        let grad_branch_lengths_flat: Vec<f64> = grad_branch_lengths.into_iter().flatten().collect();

        let grad_branch_lengths_tensor = candle_core::Tensor::from_vec(grad_branch_lengths_flat, &candle_core::Shape::from(&[L, num_branch]), &candle_core::Device::Cpu).unwrap();


        let grad_2 = grad_res.unsqueeze(1).unwrap().unsqueeze(2).unwrap();
        let grad_1 = grad_res.unsqueeze(1).unwrap();


        Ok((
            Some(grad_2.broadcast_mul(&grad_s_tensor).unwrap()),
            Some(grad_1.broadcast_mul(&grad_sqrt_pi_tensor).unwrap()),
            Some(grad_1.broadcast_mul(&grad_branch_lengths_tensor).unwrap().sum(0).unwrap()),
        ))

    }
}

impl candle_core::CustomOp3 for FelsensteinWithEdgeFwdOp {

    fn name(&self) -> &'static str {
        "FelsensteinWithEdgeFwdOp"
    }

    /// S, sqrt
    fn cpu_fwd(
            &self,
            s1: &candle_core::CpuStorage,
            l1: &candle_core::Layout,
            s2: &candle_core::CpuStorage,
            l2: &candle_core::Layout,
            s3: &candle_core::CpuStorage,
            l3: &candle_core::Layout,
        ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        
        let s = tensor2nalgebra2d(s1, l1);
        let sqrt_pi = tensor2nalgebra1d(s2, l2);
        let branch_lengths = tensor2vec(s3, l3);

        let mut felsenstein = self.felsenstein.lock().unwrap();
        let likelihood = felsenstein.calculate_likelihoods_with_branch_lengths(&s, &sqrt_pi, &branch_lengths);

        let shape = candle_core::Shape::from(&[likelihood.len()]);
        let cpu_storage = candle_core::CpuStorage::F64(likelihood);
        return Ok((cpu_storage, shape))
    }

    fn bwd(
            &self,
            _arg1: &candle_core::Tensor,
            _arg2: &candle_core::Tensor,
            _arg3: &candle_core::Tensor,
            _res: &candle_core::Tensor,
            _grad_res: &candle_core::Tensor,
        ) -> candle_core::Result<(Option<candle_core::Tensor>, Option<candle_core::Tensor>, Option<candle_core::Tensor>)> {
        panic!("FelsensteinWithEdgeFwdOp does not support backward pass");
    }
}
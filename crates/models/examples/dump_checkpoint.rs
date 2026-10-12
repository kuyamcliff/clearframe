//! Print tensor names, shapes and sums for a checkpoint (developer tool).
fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_checkpoint <file>");
    let sd = clearframe_models::torchload::load_checkpoint(std::path::Path::new(&path), clearframe_models::torchload::DEFAULT_WRAPPERS)
        .expect("load");
    for (n, t) in &sd.tensors {
        let s: f64 = t.data.iter().map(|v| *v as f64).sum();
        println!("{n} {:?} {:.6e}", t.shape, s);
    }
}

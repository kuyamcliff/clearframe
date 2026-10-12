//! Convert a checkpoint to safetensors (developer tool; the app uses the installer).
//! usage: convert_checkpoint <src> <dst> [strip_prefix] [drop_prefix,...]
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rules = clearframe_models::convert::KeyRules {
        strip_prefixes: a.get(3).map(|s| vec![s.clone()]).unwrap_or_default(),
        drop_prefixes: a.get(4).map(|s| s.split(',').map(String::from).collect()).unwrap_or_default(),
        wrappers: vec![],
    };
    let n = clearframe_models::convert::checkpoint_to_safetensors(a[1].as_ref(), a[2].as_ref(), &rules).expect("convert");
    println!("{} tensors -> {}", n, a[2]);
}

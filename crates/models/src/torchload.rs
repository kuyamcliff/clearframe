//! Safe loader for PyTorch checkpoint files (`.pth` / `.pkl`).
//!
//! PyTorch checkpoints are Python pickles. Loading them with Python's `pickle`
//! module can execute arbitrary code, so ClipFX Enhance never does that. Instead this
//! module implements a small, *restricted* pickle virtual machine:
//!
//! * only the opcodes needed to describe nested dicts/lists/tuples of tensors are
//!   supported;
//! * `GLOBAL` references are resolved against a fixed allow-list (tensor rebuild
//!   helpers, typed storage classes and `collections.OrderedDict`). Any other
//!   global makes loading fail;
//! * nothing is ever "called" — `REDUCE` on an allow-listed global is interpreted
//!   by Rust code that only constructs data.
//!
//! Both the modern zip-based format (`torch.save` with `_use_new_zipfile_serialization`)
//! and the legacy sequential format are supported.

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid checkpoint archive: {0}")]
    Zip(String),
    #[error("malformed pickle: {0}")]
    Pickle(String),
    #[error("checkpoint references a disallowed Python object `{0}`; refusing to load")]
    Disallowed(String),
    #[error("unsupported tensor data type `{0}`")]
    DType(String),
    #[error("checkpoint does not contain a tensor dictionary")]
    NoStateDict,
}

type Result<T> = std::result::Result<T, LoadError>;

/// Storage element types we understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageType {
    F32,
    F16,
    BF16,
    F64,
    I64,
    I32,
    U8,
    Bool,
}

impl StorageType {
    fn from_class(name: &str) -> Option<Self> {
        Some(match name {
            "FloatStorage" => Self::F32,
            "HalfStorage" => Self::F16,
            "BFloat16Storage" => Self::BF16,
            "DoubleStorage" => Self::F64,
            "LongStorage" => Self::I64,
            "IntStorage" => Self::I32,
            "ByteStorage" => Self::U8,
            "BoolStorage" => Self::Bool,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 | Self::I64 => 8,
            Self::U8 | Self::Bool => 1,
        }
    }
}

/// Reference to (part of) a storage blob.
#[derive(Debug, Clone)]
pub struct StorageRef {
    pub key: String,
    pub dtype: StorageType,
    /// Element offset into the root storage (legacy "view" storages only).
    pub view_offset: usize,
}

#[derive(Debug, Clone)]
pub struct TensorRef {
    pub storage: StorageRef,
    pub offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

/// Values produced by the restricted pickle machine.
#[derive(Debug, Clone)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage(StorageRef),
    Tensor(TensorRef),
    Mark,
}

impl Value {
    fn as_int(&self) -> Result<i64> {
        match self {
            Value::Int(v) => Ok(*v),
            Value::Bool(b) => Ok(*b as i64),
            other => Err(LoadError::Pickle(format!("expected int, got {other:?}"))),
        }
    }
    fn as_usize_list(&self) -> Result<Vec<usize>> {
        match self {
            Value::Tuple(v) | Value::List(v) => v.iter().map(|x| x.as_int().and_then(|i| usize::try_from(i).map_err(|_| LoadError::Pickle("negative dim".into())))).collect(),
            other => Err(LoadError::Pickle(format!("expected tuple of ints, got {other:?}"))),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Upper bounds that defend against malicious or corrupt pickles.
const MAX_STACK: usize = 1 << 20;
const MAX_STRING: usize = 64 << 20;
const MAX_DIMS: usize = 8;

/// Allow-listed globals. Everything else is rejected.
fn check_global(module: &str, name: &str) -> Result<()> {
    let ok = match module {
        "torch._utils" => matches!(name, "_rebuild_tensor_v2" | "_rebuild_tensor" | "_rebuild_parameter"),
        "torch" => StorageType::from_class(name).is_some(),
        "collections" => name == "OrderedDict",
        "torch.nn.modules.container" | "builtins" => false,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(LoadError::Disallowed(format!("{module}.{name}")))
    }
}

struct Machine<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: HashMap<u32, Value>,
    /// Mark positions in `stack`.
    marks: Vec<usize>,
}

impl<'a> Machine<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, stack: Vec::new(), memo: HashMap::new(), marks: Vec::new() }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.data.len().saturating_sub(self.pos) {
            return Err(LoadError::Pickle("unexpected end of data".into()));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
            if self.pos - start > 4096 {
                return Err(LoadError::Pickle("line too long".into()));
            }
        }
        let s = String::from_utf8_lossy(&self.data[start..self.pos]).into_owned();
        self.pos += 1;
        Ok(s)
    }
    fn push(&mut self, v: Value) -> Result<()> {
        if self.stack.len() >= MAX_STACK {
            return Err(LoadError::Pickle("stack overflow".into()));
        }
        self.stack.push(v);
        Ok(())
    }
    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().ok_or_else(|| LoadError::Pickle("stack underflow".into()))
    }
    fn top(&mut self) -> Result<&mut Value> {
        self.stack.last_mut().ok_or_else(|| LoadError::Pickle("empty stack".into()))
    }
    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let m = self.marks.pop().ok_or_else(|| LoadError::Pickle("mark not found".into()))?;
        if m > self.stack.len() {
            return Err(LoadError::Pickle("bad mark".into()));
        }
        Ok(self.stack.split_off(m))
    }
    fn string(&mut self, n: usize) -> Result<String> {
        if n > MAX_STRING {
            return Err(LoadError::Pickle("string too large".into()));
        }
        let b = self.take(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    fn persistent_load(&self, pid: Value) -> Result<Value> {
        let items = match pid {
            Value::Tuple(items) => items,
            other => return Err(LoadError::Pickle(format!("unexpected persistent id {other:?}"))),
        };
        if items.first().and_then(|v| v.as_str()) != Some("storage") || items.len() < 5 {
            return Err(LoadError::Pickle("unsupported persistent id".into()));
        }
        let dtype = match &items[1] {
            Value::Global(m, n) if m == "torch" => StorageType::from_class(n).ok_or_else(|| LoadError::DType(n.clone()))?,
            other => return Err(LoadError::DType(format!("{other:?}"))),
        };
        let key = match &items[2] {
            Value::Str(s) => s.clone(),
            Value::Int(i) => i.to_string(),
            other => return Err(LoadError::Pickle(format!("bad storage key {other:?}"))),
        };
        // Legacy view metadata: (view_key, offset, view_size)
        let mut view_offset = 0usize;
        if let Some(Value::Tuple(view)) = items.get(5) {
            if view.len() >= 2 {
                view_offset = view[1].as_int()? as usize;
            }
        }
        Ok(Value::Storage(StorageRef { key, dtype, view_offset }))
    }

    fn reduce(&mut self, func: Value, args: Value) -> Result<Value> {
        let (module, name) = match func {
            Value::Global(m, n) => (m, n),
            other => return Err(LoadError::Pickle(format!("REDUCE on non-global {other:?}"))),
        };
        let args = match args {
            Value::Tuple(a) => a,
            other => return Err(LoadError::Pickle(format!("REDUCE args not a tuple: {other:?}"))),
        };
        match (module.as_str(), name.as_str()) {
            ("collections", "OrderedDict") => Ok(Value::Dict(Vec::new())),
            ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => {
                if args.len() < 4 {
                    return Err(LoadError::Pickle("_rebuild_tensor: too few args".into()));
                }
                let storage = match &args[0] {
                    Value::Storage(s) => s.clone(),
                    other => return Err(LoadError::Pickle(format!("tensor without storage: {other:?}"))),
                };
                let offset = args[1].as_int()? as usize;
                let shape = args[2].as_usize_list()?;
                let stride = args[3].as_usize_list()?;
                if shape.len() != stride.len() || shape.len() > MAX_DIMS {
                    return Err(LoadError::Pickle("bad tensor shape/stride".into()));
                }
                Ok(Value::Tensor(TensorRef { storage, offset, shape, stride }))
            }
            ("torch._utils", "_rebuild_parameter") => args.into_iter().next().ok_or_else(|| LoadError::Pickle("_rebuild_parameter: no data".into())),
            _ => Err(LoadError::Disallowed(format!("{module}.{name}"))),
        }
    }

    fn run(&mut self) -> Result<Value> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    self.u8()?; // PROTO
                }
                0x95 => {
                    self.take(8)?; // FRAME
                }
                b'.' => return self.pop(), // STOP
                b'(' => self.marks.push(self.stack.len()),
                b'N' => self.push(Value::None)?,
                0x88 => self.push(Value::Bool(true))?,
                0x89 => self.push(Value::Bool(false))?,
                b'J' => {
                    let v = i32::from_le_bytes(self.take(4)?.try_into().unwrap());
                    self.push(Value::Int(v as i64))?
                }
                b'K' => {
                    let v = self.u8()?;
                    self.push(Value::Int(v as i64))?
                }
                b'M' => {
                    let v = self.u16()?;
                    self.push(Value::Int(v as i64))?
                }
                0x8a => {
                    // LONG1
                    let n = self.u8()? as usize;
                    let b = self.take(n)?;
                    if n > 8 {
                        // Values this large only appear as the legacy magic number; keep low bits.
                        let mut v: i64 = 0;
                        for (i, byte) in b.iter().take(8).enumerate() {
                            v |= (*byte as i64) << (8 * i);
                        }
                        self.push(Value::Int(v))?
                    } else {
                        let mut v: i64 = 0;
                        for (i, byte) in b.iter().enumerate() {
                            v |= (*byte as i64) << (8 * i);
                        }
                        if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                            v -= 1i64 << (8 * n);
                        }
                        self.push(Value::Int(v))?
                    }
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.push(Value::Float(v))?
                }
                b'X' => {
                    let n = self.u32()? as usize;
                    let s = self.string(n)?;
                    self.push(Value::Str(s))?
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    let s = self.string(n)?;
                    self.push(Value::Str(s))?
                }
                b'T' => {
                    let n = self.u32()? as usize;
                    let s = self.string(n)?;
                    self.push(Value::Str(s))?
                }
                b'U' => {
                    let n = self.u8()? as usize;
                    let s = self.string(n)?;
                    self.push(Value::Str(s))?
                }
                b'B' => {
                    let n = self.u32()? as usize;
                    if n > MAX_STRING {
                        return Err(LoadError::Pickle("bytes too large".into()));
                    }
                    let b = self.take(n)?.to_vec();
                    self.push(Value::Bytes(b))?
                }
                b'C' => {
                    let n = self.u8()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.push(Value::Bytes(b))?
                }
                b']' => self.push(Value::List(Vec::new()))?,
                b'}' => self.push(Value::Dict(Vec::new()))?,
                b')' => self.push(Value::Tuple(Vec::new()))?,
                b't' => {
                    let items = self.pop_mark()?;
                    self.push(Value::Tuple(items))?
                }
                0x85 => {
                    let a = self.pop()?;
                    self.push(Value::Tuple(vec![a]))?
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Tuple(vec![a, b]))?
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Tuple(vec![a, b, c]))?
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.push(Value::List(items))?
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    let mut d = Vec::new();
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        d.push((k, v));
                    }
                    self.push(Value::Dict(d))?
                }
                b'q' => {
                    let k = self.u8()? as u32;
                    let v = self.top()?.clone();
                    self.memo.insert(k, v);
                }
                b'r' => {
                    let k = self.u32()?;
                    let v = self.top()?.clone();
                    self.memo.insert(k, v);
                }
                0x94 => {
                    let k = self.memo.len() as u32;
                    let v = self.top()?.clone();
                    self.memo.insert(k, v);
                }
                b'h' => {
                    let k = self.u8()? as u32;
                    let v = self.memo.get(&k).cloned().ok_or_else(|| LoadError::Pickle("memo miss".into()))?;
                    self.push(v)?
                }
                b'j' => {
                    let k = self.u32()?;
                    let v = self.memo.get(&k).cloned().ok_or_else(|| LoadError::Pickle("memo miss".into()))?;
                    self.push(v)?
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    check_global(&module, &name)?;
                    self.push(Value::Global(module, name))?
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (m, n) = match (module, name) {
                        (Value::Str(m), Value::Str(n)) => (m, n),
                        _ => return Err(LoadError::Pickle("bad STACK_GLOBAL".into())),
                    };
                    check_global(&m, &n)?;
                    self.push(Value::Global(m, n))?
                }
                b'Q' => {
                    let pid = self.pop()?;
                    let v = self.persistent_load(pid)?;
                    self.push(v)?
                }
                b'R' => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let v = self.reduce(func, args)?;
                    self.push(v)?
                }
                0x81 => {
                    // NEWOBJ: cls.__new__(cls, *args)
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let v = self.reduce(cls, args)?;
                    self.push(v)?
                }
                b'b' => {
                    // BUILD: state is ignored for allow-listed objects (e.g. OrderedDict._metadata).
                    self.pop()?;
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.top()? {
                        Value::Dict(d) => d.push((k, v)),
                        _ => return Err(LoadError::Pickle("SETITEM on non-dict".into())),
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    match self.top()? {
                        Value::Dict(d) => {
                            let mut it = items.into_iter();
                            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                                d.push((k, v));
                            }
                        }
                        _ => return Err(LoadError::Pickle("SETITEMS on non-dict".into())),
                    }
                }
                b'a' => {
                    let v = self.pop()?;
                    match self.top()? {
                        Value::List(l) => l.push(v),
                        _ => return Err(LoadError::Pickle("APPEND on non-list".into())),
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    match self.top()? {
                        Value::List(l) => l.extend(items),
                        _ => return Err(LoadError::Pickle("APPENDS on non-list".into())),
                    }
                }
                b'0' => {
                    self.pop()?;
                }
                b'2' => {
                    let v = self.top()?.clone();
                    self.push(v)?
                }
                other => return Err(LoadError::Pickle(format!("unsupported opcode 0x{other:02x}"))),
            }
        }
    }
}

/// Run a restricted unpickle of `data`, returning the value and number of bytes consumed.
pub fn unpickle(data: &[u8]) -> Result<(Value, usize)> {
    let mut m = Machine::new(data);
    let v = m.run()?;
    Ok((v, m.pos))
}

/// A dense f32 tensor loaded from a checkpoint.
#[derive(Debug, Clone)]
pub struct DenseTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// An ordered list of named tensors.
#[derive(Debug, Default)]
pub struct StateDict {
    pub tensors: Vec<(String, DenseTensor)>,
}

impl StateDict {
    pub fn get(&self, name: &str) -> Option<&DenseTensor> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
    pub fn total_params(&self) -> usize {
        self.tensors.iter().map(|(_, t)| t.data.len()).sum()
    }
}

fn decode_storage(bytes: &[u8], dtype: StorageType) -> Vec<f32> {
    match dtype {
        StorageType::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        StorageType::F64 => bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        StorageType::F16 => bytes.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        StorageType::BF16 => bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        StorageType::I64 => bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        StorageType::I32 => bytes.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        StorageType::U8 | StorageType::Bool => bytes.iter().map(|b| *b as f32).collect(),
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

fn materialize(t: &TensorRef, storage: &[f32]) -> Result<DenseTensor> {
    let numel: usize = t.shape.iter().product();
    let base = t.storage.view_offset + t.offset;
    let mut out = Vec::with_capacity(numel);
    if numel == 0 {
        return Ok(DenseTensor { shape: t.shape.clone(), data: out });
    }
    // Bounds check: max index reachable.
    let max_index = base + t.shape.iter().zip(&t.stride).map(|(d, s)| (d - 1) * s).sum::<usize>();
    if max_index >= storage.len() {
        return Err(LoadError::Pickle("tensor exceeds storage bounds".into()));
    }
    let nd = t.shape.len();
    let mut idx = vec![0usize; nd];
    for _ in 0..numel {
        let off: usize = base + idx.iter().zip(&t.stride).map(|(i, s)| i * s).sum::<usize>();
        out.push(storage[off]);
        for d in (0..nd).rev() {
            idx[d] += 1;
            if idx[d] < t.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(DenseTensor { shape: t.shape.clone(), data: out })
}

/// Collect `(name, tensor)` pairs from a nested dict value. Nested dicts are
/// flattened with `.` separators.
fn collect_tensors(v: &Value, prefix: &str, out: &mut Vec<(String, TensorRef)>) {
    if let Value::Dict(items) = v {
        for (k, val) in items {
            let key = match k {
                Value::Str(s) => s.clone(),
                Value::Int(i) => i.to_string(),
                _ => continue,
            };
            let name = if prefix.is_empty() { key } else { format!("{prefix}.{key}") };
            match val {
                Value::Tensor(t) => out.push((name, t.clone())),
                Value::Dict(_) => collect_tensors(val, &name, out),
                _ => {}
            }
        }
    }
}

/// Pick the actual weights dict out of a checkpoint. Many training frameworks wrap
/// the state dict, e.g. `{"params_ema": {...}}` (BasicSR) or `{"state_dict": {...}}`.
fn select_root<'v>(v: &'v Value, preferred: &[&str]) -> &'v Value {
    if let Value::Dict(items) = v {
        for key in preferred {
            for (k, val) in items {
                if k.as_str() == Some(key) {
                    if let Value::Dict(_) = val {
                        return val;
                    }
                }
            }
        }
    }
    v
}

/// Default wrapper keys, in priority order.
pub const DEFAULT_WRAPPERS: &[&str] = &["params_ema", "params", "state_dict", "model", "net_g"];

/// Load a checkpoint from bytes.
pub fn load_checkpoint_bytes(bytes: &[u8], wrappers: &[&str]) -> Result<StateDict> {
    if bytes.len() >= 4 && &bytes[..4] == b"PK\x03\x04" {
        load_zip(bytes, wrappers)
    } else {
        load_legacy(bytes, wrappers)
    }
}

/// Load a checkpoint from a file path.
pub fn load_checkpoint(path: &std::path::Path, wrappers: &[&str]) -> Result<StateDict> {
    let bytes = std::fs::read(path)?;
    load_checkpoint_bytes(&bytes, wrappers)
}

fn load_zip(bytes: &[u8], wrappers: &[&str]) -> Result<StateDict> {
    let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| LoadError::Zip(e.to_string()))?;
    let pkl_name = (0..zip.len())
        .filter_map(|i| zip.name_for_index(i).map(|s| s.to_string()))
        .find(|n| n.ends_with("/data.pkl") || n == "data.pkl")
        .ok_or_else(|| LoadError::Zip("no data.pkl".into()))?;
    let prefix = pkl_name.trim_end_matches("data.pkl").to_string();
    let pkl = read_zip_entry(&mut zip, &pkl_name, MAX_STRING * 4)?;
    let (root, _) = unpickle(&pkl)?;
    let root = select_root(&root, wrappers);
    let mut refs = Vec::new();
    collect_tensors(root, "", &mut refs);
    if refs.is_empty() {
        return Err(LoadError::NoStateDict);
    }
    let mut cache: HashMap<String, Vec<f32>> = HashMap::new();
    let mut tensors = Vec::with_capacity(refs.len());
    for (name, t) in refs {
        if !cache.contains_key(&t.storage.key) {
            if !t.storage.key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(LoadError::Zip("invalid storage key".into()));
            }
            let entry = format!("{prefix}data/{}", t.storage.key);
            let raw = read_zip_entry(&mut zip, &entry, usize::MAX)?;
            cache.insert(t.storage.key.clone(), decode_storage(&raw, t.storage.dtype));
        }
        let dense = materialize(&t, &cache[&t.storage.key])?;
        tensors.push((name, dense));
    }
    Ok(StateDict { tensors })
}

fn read_zip_entry<R: Read + Seek>(zip: &mut zip::ZipArchive<R>, name: &str, limit: usize) -> Result<Vec<u8>> {
    let mut f = zip.by_name(name).map_err(|e| LoadError::Zip(format!("{name}: {e}")))?;
    let size = f.size() as usize;
    if size > limit {
        return Err(LoadError::Zip(format!("{name} too large")));
    }
    let mut buf = Vec::with_capacity(size);
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

fn load_legacy(bytes: &[u8], wrappers: &[&str]) -> Result<StateDict> {
    let mut pos = 0usize;
    let next = |pos: &mut usize| -> Result<Value> {
        let (v, n) = unpickle(&bytes[*pos..])?;
        *pos += n;
        Ok(v)
    };
    let magic = next(&mut pos)?;
    // 0x1950a86a20f9469cfc6c truncated to the low 64 bits.
    if !matches!(magic, Value::Int(v) if v == 0x50a86a20f9469cfc6c_u128 as i64) {
        return Err(LoadError::Pickle("not a PyTorch checkpoint (bad magic number)".into()));
    }
    let _protocol = next(&mut pos)?;
    let _sys_info = next(&mut pos)?;
    let root = next(&mut pos)?;
    let keys = next(&mut pos)?;
    let keys: Vec<String> = match keys {
        Value::List(l) => l
            .into_iter()
            .map(|k| match k {
                Value::Str(s) => Ok(s),
                other => Err(LoadError::Pickle(format!("bad storage key {other:?}"))),
            })
            .collect::<Result<_>>()?,
        _ => return Err(LoadError::Pickle("missing storage key list".into())),
    };

    let root_sel = select_root(&root, wrappers);
    let mut refs = Vec::new();
    collect_tensors(root_sel, "", &mut refs);
    if refs.is_empty() {
        return Err(LoadError::NoStateDict);
    }
    // dtype for every storage key
    let mut dtypes: HashMap<String, StorageType> = HashMap::new();
    let mut all_refs = Vec::new();
    collect_tensors(&root, "", &mut all_refs);
    for (_, t) in &all_refs {
        dtypes.insert(t.storage.key.clone(), t.storage.dtype);
    }
    let mut storages: HashMap<String, Vec<f32>> = HashMap::new();
    for key in keys {
        if pos + 8 > bytes.len() {
            return Err(LoadError::Pickle("truncated storage data".into()));
        }
        let numel = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let dtype = *dtypes.get(&key).unwrap_or(&StorageType::F32);
        let nbytes = numel.checked_mul(dtype.size()).ok_or_else(|| LoadError::Pickle("storage size overflow".into()))?;
        if pos + nbytes > bytes.len() {
            return Err(LoadError::Pickle("truncated storage data".into()));
        }
        storages.insert(key, decode_storage(&bytes[pos..pos + nbytes], dtype));
        pos += nbytes;
    }
    let mut tensors = Vec::with_capacity(refs.len());
    for (name, t) in refs {
        let st = storages.get(&t.storage.key).ok_or_else(|| LoadError::Pickle(format!("missing storage {}", t.storage.key)))?;
        tensors.push((name, materialize(&t, st)?));
    }
    Ok(StateDict { tensors })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_disallowed_global() {
        // cos\nsystem\n(S'echo pwned'\ntR.
        let evil = b"cos\nsystem\n(S'echo pwned'\ntR.";
        let err = unpickle(evil).unwrap_err();
        assert!(matches!(err, LoadError::Disallowed(_)), "{err:?}");
    }

    #[test]
    fn rejects_builtins_eval_stack_global() {
        // proto 4: SHORT_BINUNICODE builtins, SHORT_BINUNICODE eval, STACK_GLOBAL
        let mut p = vec![0x80, 4];
        p.push(0x8c);
        p.push(8);
        p.extend_from_slice(b"builtins");
        p.push(0x8c);
        p.push(4);
        p.extend_from_slice(b"eval");
        p.push(0x93);
        p.push(b'.');
        assert!(matches!(unpickle(&p).unwrap_err(), LoadError::Disallowed(_)));
    }

    #[test]
    fn truncated_input_is_an_error() {
        assert!(unpickle(&[0x80, 2, b'X', 255, 255, 0, 0]).is_err());
        assert!(unpickle(&[]).is_err());
    }

    #[test]
    fn parses_simple_dict() {
        // }q\x00(X\x01\x00\x00\x00aK\x01u.
        let p = b"}q\x00(X\x01\x00\x00\x00aK\x01u.";
        let (v, _) = unpickle(p).unwrap();
        match v {
            Value::Dict(d) => {
                assert_eq!(d.len(), 1);
                assert!(matches!(d[0].1, Value::Int(1)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn half_conversion() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert!((half_to_f32(0x3555) - 0.333_251_95).abs() < 1e-6);
    }

    #[test]
    fn strided_materialize() {
        // 2x3 tensor stored transposed (stride 1,2)
        let t = TensorRef { storage: StorageRef { key: "0".into(), dtype: StorageType::F32, view_offset: 0 }, offset: 0, shape: vec![2, 3], stride: vec![1, 2] };
        let st = vec![0., 1., 2., 3., 4., 5.];
        let d = materialize(&t, &st).unwrap();
        assert_eq!(d.data, vec![0., 2., 4., 1., 3., 5.]);
        let bad = TensorRef { shape: vec![4, 3], ..t };
        assert!(materialize(&bad, &st).is_err());
    }
}

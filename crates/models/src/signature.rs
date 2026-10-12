//! Ed25519 signatures for remote catalog documents.

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const BUNDLED_TRUSTED_KEYS: &str = include_str!("../../../model-catalog/trusted_keys.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustedKey {
    pub id: String,
    pub public_key_base64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrustedKeys {
    #[serde(default)]
    pub comment: String,
    pub keys: Vec<TrustedKey>,
}

impl TrustedKeys {
    pub fn bundled() -> Self {
        serde_json::from_str(BUNDLED_TRUSTED_KEYS).unwrap_or_default()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SignatureError {
    #[error("no trusted catalog signing key is configured in this build")]
    NoKeys,
    #[error("the signature is malformed")]
    Malformed,
    #[error("the catalog signature does not match any trusted key")]
    Untrusted,
}

/// Verify `data` against a base64 signature using any trusted key; returns the key id.
pub fn verify(data: &[u8], signature_b64: &str, keys: &TrustedKeys) -> Result<String, SignatureError> {
    if keys.keys.is_empty() {
        return Err(SignatureError::NoKeys);
    }
    let sig_bytes = B64.decode(signature_b64.trim()).map_err(|_| SignatureError::Malformed)?;
    let sig = Signature::from_slice(&sig_bytes).map_err(|_| SignatureError::Malformed)?;
    for k in &keys.keys {
        let Ok(pk) = B64.decode(k.public_key_base64.trim()) else { continue };
        let Ok(pk): Result<[u8; 32], _> = pk.try_into() else { continue };
        let Ok(vk) = VerifyingKey::from_bytes(&pk) else { continue };
        if vk.verify(data, &sig).is_ok() {
            return Ok(k.id.clone());
        }
    }
    Err(SignatureError::Untrusted)
}

/// Generate a key pair: (secret_base64, public_base64).
pub fn keygen() -> (String, String) {
    let mut seed = [0u8; 32];
    getrandom_fill(&mut seed);
    let sk = SigningKey::from_bytes(&seed);
    (B64.encode(sk.to_bytes()), B64.encode(sk.verifying_key().to_bytes()))
}

/// Sign `data` with a base64 secret key.
pub fn sign(data: &[u8], secret_b64: &str) -> Result<String, SignatureError> {
    let sk = B64.decode(secret_b64.trim()).map_err(|_| SignatureError::Malformed)?;
    let sk: [u8; 32] = sk.try_into().map_err(|_| SignatureError::Malformed)?;
    Ok(B64.encode(SigningKey::from_bytes(&sk).sign(data).to_bytes()))
}

fn getrandom_fill(buf: &mut [u8]) {
    use rand_core::RngCore;
    rand_core::OsRng.fill_bytes(buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify() {
        let (sk, pk) = keygen();
        let keys = TrustedKeys { comment: String::new(), keys: vec![TrustedKey { id: "test".into(), public_key_base64: pk }] };
        let sig = sign(b"catalog", &sk).unwrap();
        assert_eq!(verify(b"catalog", &sig, &keys).unwrap(), "test");
        assert!(matches!(verify(b"tampered", &sig, &keys), Err(SignatureError::Untrusted)));
        assert!(matches!(verify(b"catalog", "!!", &keys), Err(SignatureError::Malformed)));
        assert!(matches!(verify(b"catalog", &sig, &TrustedKeys::default()), Err(SignatureError::NoKeys)));
    }
}

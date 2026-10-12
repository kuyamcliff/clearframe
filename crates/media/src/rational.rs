//! Exact rational numbers for frame rates and time bases.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

impl Rational {
    pub const fn new_raw(num: i64, den: i64) -> Self {
        Self { num, den }
    }

    pub fn new(num: i64, den: i64) -> Self {
        if den == 0 {
            return Self { num: 0, den: 1 };
        }
        let g = gcd(num, den);
        let s = if den < 0 { -1 } else { 1 };
        Self { num: s * num / g, den: s * den / g }
    }

    /// Parse `"30000/1001"`, `"25"` or `"29.97"`.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if let Some((a, b)) = s.split_once('/') {
            let n = a.trim().parse::<i64>().ok()?;
            let d = b.trim().parse::<i64>().ok()?;
            if d == 0 {
                return None;
            }
            return Some(Self::new(n, d));
        }
        if let Ok(n) = s.parse::<i64>() {
            return Some(Self::new(n, 1));
        }
        let f = s.parse::<f64>().ok()?;
        Some(Self::from_f64(f))
    }

    /// Approximate a float, snapping to common NTSC rates.
    pub fn from_f64(f: f64) -> Self {
        for (n, d) in [(24000, 1001), (30000, 1001), (60000, 1001), (120000, 1001)] {
            if (f - n as f64 / d as f64).abs() < 0.0005 {
                return Self::new(n, d);
            }
        }
        let den = 1000i64;
        Self::new((f * den as f64).round() as i64, den)
    }

    pub fn as_f64(&self) -> f64 {
        if self.den == 0 {
            0.0
        } else {
            self.num as f64 / self.den as f64
        }
    }

    pub fn is_valid(&self) -> bool {
        self.num > 0 && self.den > 0
    }

    pub fn mul(&self, o: Rational) -> Rational {
        Rational::new(self.num * o.num, self.den * o.den)
    }

    pub fn div(&self, o: Rational) -> Rational {
        Rational::new(self.num * o.den, self.den * o.num)
    }

    pub fn mul_int(&self, k: i64) -> Rational {
        Rational::new(self.num * k, self.den)
    }

    /// FFmpeg-style string.
    pub fn to_ffmpeg(&self) -> String {
        format!("{}/{}", self.num, self.den)
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{:.3}", self.as_f64())
        }
    }
}

/// Number of frames in `duration` seconds at `fps`, rounded.
pub fn frames_for(duration: f64, fps: Rational) -> u64 {
    (duration * fps.as_f64()).round().max(0.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_reduce() {
        assert_eq!(Rational::parse("60/2").unwrap(), Rational::new(30, 1));
        assert_eq!(Rational::parse("30000/1001").unwrap(), Rational::new(30000, 1001));
        assert_eq!(Rational::parse("29.97").unwrap(), Rational::new(30000, 1001));
        assert_eq!(Rational::parse("25").unwrap(), Rational::new(25, 1));
        assert!(Rational::parse("1/0").is_none());
        assert!(Rational::parse("abc").is_none());
    }

    #[test]
    fn arithmetic() {
        let ntsc = Rational::new(24000, 1001);
        assert_eq!(ntsc.mul_int(2), Rational::new(48000, 1001));
        assert_eq!(Rational::new(60, 1).div(Rational::new(24, 1)), Rational::new(5, 2));
        assert_eq!(frames_for(2.0, Rational::new(30, 1)), 60);
        assert_eq!(Rational::new(30, 1).to_string(), "30");
        assert_eq!(ntsc.to_string(), "23.976");
    }
}

//! Statistical distributions module
//!
//! Implements Gamma distribution, Chi-square distribution, F distribution and other statistical distributions

use anyhow::{Error, bail};
use std::f64;

/// Machine precision
const EPS: f64 = 2.22045e-16;
/// Minimum floating point number
const MIN: f64 = 2.22507e-308;
const FPMIN: f64 = MIN / EPS;

/// Constants related to Gamma function
const ASWITCH: f64 = 100.0;

/// Gamma function coefficients (from Numerical Recipes)
const GAMMA_COF: [f64; 14] = [
    57.156_235_665_862_92,
    -59.597_960_355_475_49,
    14.136_097_974_741_746,
    -0.491_913_816_097_620_2,
    0.000_033_994_649_984_811_89,
    0.000_046_523_628_927_048_58,
    -0.000_098_374_475_304_879_56,
    0.000_158_088_703_224_912_5,
    -0.000_210_264_441_724_104_88,
    0.000_217_439_618_115_212_65,
    -0.000_164_318_106_536_763_9,
    0.000_084_418_223_983_852_75,
    -0.000_026_190_838_401_581_408,
    0.000_003_689_991_865_959_163,
];

/// Gauss-Legendre quadrature points (18 points)
const GAULEG_Y: [f64; 18] = [
    0.002_169_537_515_914_199_4,
    0.011_413_521_097_787_704,
    0.027_972_308_950_302_116,
    0.051_727_015_600_492_42,
    0.082_502_225_484_340_94,
    0.120_070_199_109_602_93,
    0.164_152_833_007_524_7,
    0.214_423_769_867_793_55,
    0.270_510_828_406_443_36,
    0.331_998_763_414_478_87,
    0.398_432_341_864_019_43,
    0.469_319_714_073_754_83,
    0.544_136_055_566_579_7,
    0.622_327_452_880_310_8,
    0.703_315_004_655_971_7,
    0.786_499_107_683_134_5,
    0.871_263_896_190_615_2,
    0.956_981_801_526_291_4,
];

const GAULEG_W: [f64; 18] = [
    0.005_565_719_664_244_557,
    0.012_915_947_284_065_42,
    0.020_181_515_297_735_382,
    0.027_298_621_498_568_734,
    0.034_213_810_770_299_54,
    0.040_875_750_923_643_26,
    0.047_235_083_490_265_58,
    0.053_244_713_977_759_69,
    0.058_860_144_245_324_8,
    0.064_039_797_355_015_48,
    0.068_745_323_835_736_41,
    0.072_941_885_005_653_09,
    0.076_598_410_645_870_64,
    0.079_687_828_912_071_67,
    0.082_187_266_704_339_7,
    0.084_078_218_979_661_95,
    0.085_346_685_739_338_72,
    0.085_983_275_670_394_82,
];

/// Calculate ln(Gamma(x))
///
/// # Errors
///
/// Returns an error if x is not positive.
pub fn gammln(xx: f64) -> Result<f64, Error> {
    if xx <= 0.0 {
        bail!(
            "Gamma distribution error: argument must be positive, got {}",
            xx
        );
    }

    let mut y = xx;
    let x = xx;
    let tmp_val = x + 5.242_187_5; // Rational 671/128
    let tmp = (x + 0.5) * tmp_val.ln() - tmp_val;
    let mut ser = 0.999_999_999_999_997_1;

    for cofactor in &GAMMA_COF {
        y += 1.0;
        ser += cofactor / y;
    }

    Ok(tmp + (2.506_628_274_631_000_5 * ser / x).ln())
}

/// Incomplete Gamma function P(a, x)
///
/// # Errors
///
/// Returns an error if a <= 0 or x < 0.
pub fn gammp(a: f64, x: f64) -> Result<f64, Error> {
    if x < 0.0 || a <= 0.0 {
        bail!(
            "Gamma distribution error: invalid arguments: a={}, x={} (a must be > 0 and x >= 0)",
            a,
            x
        );
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if a >= ASWITCH {
        Ok(gammpapprox(a, x, 1)?)
    } else if x < a + 1.0 {
        Ok(gser(a, x)?)
    } else {
        Ok(1.0 - gcf(a, x)?)
    }
}

/// Incomplete Gamma function Q(a, x) = 1 - P(a, x)
///
/// # Errors
///
/// Returns an error if a <= 0 or x < 0.
pub fn gammq(a: f64, x: f64) -> Result<f64, Error> {
    if x < 0.0 || a <= 0.0 {
        bail!(
            "Gamma distribution error: invalid arguments: a={}, x={} (a must be > 0 and x >= 0)",
            a,
            x
        );
    }
    if x == 0.0 {
        return Ok(1.0);
    }
    if a >= ASWITCH {
        Ok(gammpapprox(a, x, 0)?)
    } else if x < a + 1.0 {
        Ok(1.0 - gser(a, x)?)
    } else {
        Ok(gcf(a, x)?)
    }
}

/// Gamma function series expansion
fn gser(a: f64, x: f64) -> Result<f64, Error> {
    let gln = gammln(a)?;
    let mut ap = a;
    let mut delta = 1.0 / a;
    let mut sum = delta;

    loop {
        ap += 1.0;
        delta *= x / ap;
        sum += delta;
        if delta.abs() < sum.abs() * EPS {
            return Ok(sum * (-x + a * x.ln() - gln).exp());
        }
    }
}

/// Gamma function continued fraction
fn gcf(a: f64, x: f64) -> Result<f64, Error> {
    let gln = gammln(a)?;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut i: i32 = 1;
    const MAX_ITER: i32 = 10000;

    loop {
        let an = -(i as f64) * ((i as f64) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= EPS {
            break;
        }
        i += 1;
        if i > MAX_ITER {
            // If we haven't converged, return an error
            bail!(
                "Gamma function continued fraction did not converge after {} iterations",
                MAX_ITER
            );
        }
    }

    Ok((-x + a * x.ln() - gln).exp() * h)
}

/// Gamma function approximation
fn gammpapprox(a: f64, x: f64, psig: i32) -> Result<f64, Error> {
    let a1 = a - 1.0;
    let lna1 = a1.ln();
    let sqrta1 = a1.sqrt();
    let gln = gammln(a)?;

    let xu = if x > a1 {
        (a1 + 11.5 * sqrta1).max(x + 6.0 * sqrta1)
    } else {
        (0.0f64).max((a1 - 7.5 * sqrta1).min(x - 5.0 * sqrta1))
    };

    let mut sum = 0.0;
    for j in 0..18 {
        let t = x + (xu - x) * GAULEG_Y[j];
        sum += GAULEG_W[j] * (-(t - a1) + a1 * (t.ln() - lna1)).exp();
    }

    let ans = sum * (xu - x) * (a1 * (lna1 - 1.0) - gln).exp();

    if psig == 1 {
        if ans > 0.0 { Ok(1.0 - ans) } else { Ok(-ans) }
    } else if ans >= 0.0 {
        Ok(ans)
    } else {
        Ok(1.0 + ans)
    }
}

/// Chi-square distribution
pub struct Chisqdist {
    nu: f64,
    fac: f64,
}

impl Chisqdist {
    /// Create a new Chi-square distribution
    ///
    /// # Errors
    ///
    /// Returns an error if degrees of freedom is not positive.
    pub fn new(nnu: f64) -> Result<Self, Error> {
        if nnu <= 0.0 {
            bail!(
                "Chi-square distribution error: degrees of freedom must be positive, got {}",
                nnu
            )
        }
        let fac = f64::consts::LN_2 * (0.5 * nnu) + gammln(0.5 * nnu)?;
        Ok(Self { nu: nnu, fac })
    }

    /// Create a new Chi-square distribution (panics on error)
    ///
    /// # Panics
    ///
    /// Panics if degrees of freedom is not positive.
    pub fn new_or_panic(nnu: f64) -> Self {
        Self::new(nnu).expect("Failed to create Chi-square distribution")
    }

    /// Probability density function
    ///
    /// # Errors
    ///
    /// Returns an error if x2 is not positive.
    pub fn pdf(&self, x2: f64) -> Result<f64, Error> {
        if x2 <= 0.0 {
            bail!(
                "Chi-square distribution error: argument must be positive, got {}",
                x2
            )
        }
        Ok((-0.5 * (x2 - (self.nu - 2.0) * x2.ln()) - self.fac).exp())
    }

    /// Cumulative distribution function
    ///
    /// # Errors
    ///
    /// Returns an error if x2 is negative.
    pub fn cdf(&self, x2: f64) -> Result<f64, Error> {
        if x2 < 0.0 {
            bail!(
                "Chi-square distribution error: argument must be non-negative, got {}",
                x2
            )
        }
        gammp(0.5 * self.nu, 0.5 * x2)
    }

    /// Inverse cumulative distribution function
    ///
    /// # Errors
    ///
    /// Returns an error if p is not in [0, 1).
    pub fn invcdf(&self, p: f64) -> Result<f64, Error> {
        if !(0.0..1.0).contains(&p) {
            bail!(
                "Chi-square distribution error: probability must be in [0, 1), got {}",
                p
            )
        }
        Ok(2.0 * invgammp(p, 0.5 * self.nu)?)
    }
}

/// Inverse incomplete Gamma function
fn invgammp(p: f64, a: f64) -> Result<f64, Error> {
    let a1 = a - 1.0;
    let eps = 1e-8;
    let gln = gammln(a)?;

    if a <= 0.0 {
        bail!(
            "Gamma distribution error: a must be positive for inverse Gamma, got {}",
            a
        )
    }
    if p >= 1.0 {
        return Ok((100.0f64).max(a + 100.0 * a.sqrt()));
    }
    if p <= 0.0 {
        return Ok(0.0);
    }

    let (mut x, afac, lna1_opt) = if a > 1.0 {
        let lna1 = a1.ln();
        let afac = (a1 * (lna1 - 1.0) - gln).exp();
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            x = -x;
        }
        let x = (1e-3f64).max(a * (1.0 - 1.0 / (9.0 * a) - x / (3.0 * a.sqrt())).powi(3));
        (x, afac, Some(lna1))
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        let x = if p < t {
            (p / t).powf(1.0 / a)
        } else {
            1.0 - (1.0 - (p - t) / (1.0 - t)).ln()
        };
        (x, 0.0, None)
    };

    for _j in 0..12 {
        if x <= 0.0 {
            return Ok(0.0);
        }
        let err = gammp(a, x)? - p;
        let t = if a > 1.0 {
            let lna1 = lna1_opt.unwrap();
            afac * (-(x - a1) + a1 * (x.ln() - lna1)).exp()
        } else {
            (-x + a1 * x.ln() - gln).exp()
        };
        let u = err / t;
        let t = u / (1.0 - 0.5 * (1.0f64).min(u * ((a - 1.0) / x - 1.0)));
        x -= t;
        if x <= 0.0 {
            x = 0.5 * (x + t);
        }
        if t.abs() < eps * x {
            break;
        }
    }

    Ok(x)
}

/// F distribution
pub struct Fdist {
    nu1: f64,
    nu2: f64,
    fac: f64,
}

impl Fdist {
    /// Create a new F distribution
    ///
    /// # Errors
    ///
    /// Returns an error if either degrees of freedom is not positive.
    pub fn new(nnu1: f64, nnu2: f64) -> Result<Self, Error> {
        if nnu1 <= 0.0 || nnu2 <= 0.0 {
            bail!(
                "F distribution error: degrees of freedom must be positive, got nu1={}, nu2={}",
                nnu1,
                nnu2
            )
        }
        let nu1 = nnu1;
        let nu2 = nnu2;
        let fac = 0.5 * (nu1 * nu1.ln() + nu2 * nu2.ln()) + gammln(0.5 * (nu1 + nu2))?
            - gammln(0.5 * nu1)?
            - gammln(0.5 * nu2)?;
        Ok(Self { nu1, nu2, fac })
    }

    /// Create a new F distribution (panics on error)
    ///
    /// # Panics
    ///
    /// Panics if either degrees of freedom is not positive.
    pub fn new_or_panic(nnu1: f64, nnu2: f64) -> Self {
        Self::new(nnu1, nnu2).expect("Failed to create F distribution")
    }

    /// Probability density function
    ///
    /// # Errors
    ///
    /// Returns an error if f is not positive.
    pub fn pdf(&self, f: f64) -> Result<f64, Error> {
        if f <= 0.0 {
            bail!("F distribution error: argument must be positive, got {}", f)
        }
        let nu1 = self.nu1;
        let nu2 = self.nu2;
        Ok(
            ((0.5 * nu1 - 1.0) * f.ln() - 0.5 * (nu1 + nu2) * (nu2 + nu1 * f).ln() + self.fac)
                .exp(),
        )
    }

    /// Cumulative distribution function
    ///
    /// # Errors
    ///
    /// Returns an error if f is negative.
    pub fn cdf(&self, f: f64) -> Result<f64, Error> {
        if f < 0.0 {
            bail!(
                "F distribution error: argument must be non-negative, got {}",
                f
            )
        }
        let nu1 = self.nu1;
        let nu2 = self.nu2;
        betai(0.5 * nu1, 0.5 * nu2, nu1 * f / (nu2 + nu1 * f))
    }

    /// Inverse cumulative distribution function
    ///
    /// # Errors
    ///
    /// Returns an error if p is not in (0, 1).
    pub fn invcdf(&self, p: f64) -> Result<f64, Error> {
        if p <= 0.0 || p >= 1.0 {
            bail!(
                "F distribution error: probability must be in (0, 1), got {}",
                p
            )
        }
        let nu1 = self.nu1;
        let nu2 = self.nu2;
        let x = invbetai(p, 0.5 * nu1, 0.5 * nu2)?;
        Ok(nu2 * x / (nu1 * (1.0 - x)))
    }
}

/// Incomplete Beta function
///
/// # Errors
///
/// Returns an error if a <= 0, b <= 0, or x not in [0, 1].
fn betai(a: f64, b: f64, x: f64) -> Result<f64, Error> {
    if a <= 0.0 || b <= 0.0 {
        bail!(
            "Beta distribution error: parameters must be positive, got a={}, b={}",
            a,
            b
        )
    }
    if !(0.0..=1.0).contains(&x) {
        bail!("Beta distribution error: x must be in [0, 1], got {}", x)
    }
    if x == 0.0 || x == 1.0 {
        return Ok(x);
    }

    const SWITCH: f64 = 3000.0;
    if a > SWITCH && b > SWITCH {
        return betaiapprox(a, b, x);
    }

    let bt = (gammln(a + b)? - gammln(a)? - gammln(b)? + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok(bt * betacf(a, b, x) / a)
    } else {
        Ok(1.0 - bt * betacf(b, a, 1.0 - x) / b)
    }
}

/// Beta function continued fraction
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..10000 {
        let m2 = 2.0 * (m as f64);
        let mut aa = (m as f64) * (b - (m as f64)) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + (m as f64)) * (qab + (m as f64)) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= EPS {
            break;
        }
    }

    h
}

/// Beta function approximation
fn betaiapprox(a: f64, b: f64, x: f64) -> Result<f64, Error> {
    let a1 = a - 1.0;
    let b1 = b - 1.0;
    let mu = a / (a + b);
    let lnmu = mu.ln();
    let lnmuc = (1.0 - mu).ln();
    let t = (a * b / ((a + b).powi(2) * (a + b + 1.0))).sqrt();

    let xu = if x > a / (a + b) {
        if x >= 1.0 {
            return Ok(1.0);
        }
        (1.0f64).min((mu + 10.0 * t).max(x + 5.0 * t))
    } else {
        if x <= 0.0 {
            return Ok(0.0);
        }
        (0.0f64).max((mu - 10.0 * t).min(x - 5.0 * t))
    };

    let mut sum = 0.0;
    for j in 0..18 {
        let t = x + (xu - x) * GAULEG_Y[j];
        sum += GAULEG_W[j] * (a1 * (t.ln() - lnmu) + b1 * ((1.0 - t).ln() - lnmuc)).exp();
    }

    let gln_a = gammln(a)?;
    let gln_b = gammln(b)?;
    let gln_ab = gammln(a + b)?;
    let ans = sum * (xu - x) * (a1 * lnmu - gln_a + b1 * lnmuc - gln_b + gln_ab).exp();

    if ans > 0.0 { Ok(1.0 - ans) } else { Ok(-ans) }
}

/// Inverse incomplete Beta function
///
/// # Errors
///
/// Returns an error if the computation fails to converge.
fn invbetai(p: f64, a: f64, b: f64) -> Result<f64, Error> {
    let a1 = a - 1.0;
    let b1 = b - 1.0;

    if p <= 0.0 {
        return Ok(0.0);
    }
    if p >= 1.0 {
        return Ok(1.0);
    }

    let mut x = if a >= 1.0 && b >= 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            x = -x;
        }
        let al = (x * x - 3.0) / 6.0;
        let h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0));
        let w = (x * (al + h).sqrt() / h)
            - (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h));
        a / (a + b * (2.0 * w).exp())
    } else {
        let lna = (a / (a + b)).ln();
        let lnb = (b / (a + b)).ln();
        let t = (a * lna).exp() / a;
        let u = (b * lnb).exp() / b;
        let w = t + u;
        if p < t / w {
            (a * w * p).powf(1.0 / a)
        } else {
            1.0 - (b * w * (1.0 - p)).powf(1.0 / b)
        }
    };

    let afac = -gammln(a)? - gammln(b)? + gammln(a + b)?;

    for j in 0..10 {
        if x == 0.0 || x == 1.0 {
            return Ok(x);
        }
        let err = betai(a, b, x)? - p;
        let t = (a1 * x.ln() + b1 * (1.0 - x).ln() + afac).exp();
        let u = err / t;
        let t = u / (1.0 - 0.5 * (1.0f64).min(u * (a1 / x - b1 / (1.0 - x))));
        x -= t;
        if x <= 0.0 {
            x = 0.5 * (x + t);
        }
        if x >= 1.0 {
            x = 0.5 * (x + t + 1.0);
        }
        if t.abs() < EPS * x && j > 0 {
            break;
        }
    }

    Ok(x)
}

/// Error function class
pub struct Erf;

impl Erf {
    const NCOF: usize = 28;
    const COF: [f64; 28] = [
        -1.302_653_719_781_709_4,
        0.641_969_792_356_490_3,
        0.019_476_473_204_185_836,
        -0.009_561_514_786_808_63,
        -9.465_953_444_820_36e-4,
        3.668_394_978_527_61e-4,
        4.252_332_480_690_7e-5,
        -2.027_857_811_253_4e-5,
        -1.624_290_004_647e-6,
        1.303_655_835_580e-6,
        1.562_644_172_2e-8,
        -8.523_809_591_5e-8,
        6.529_054_439e-9,
        5.059_343_495e-9,
        -9.913_641_56e-10,
        -2.273_651_22e-10,
        9.646_791_1e-11,
        2.394_038e-12,
        -6.886_027e-12,
        8.944_87e-13,
        3.130_92e-13,
        -1.127_08e-13,
        3.81e-16,
        7.106e-15,
        -1.523e-15,
        -9.4e-17,
        1.21e-16,
        -2.8e-17,
    ];

    pub fn new() -> Self {
        Self
    }

    /// Error function
    pub fn erf(&self, x: f64) -> f64 {
        if x >= 0.0 {
            1.0 - self.erfccheb(x)
        } else {
            self.erfccheb(-x) - 1.0
        }
    }

    /// Complementary error function
    pub fn erfc(&self, x: f64) -> f64 {
        if x >= 0.0 {
            self.erfccheb(x)
        } else {
            2.0 - self.erfccheb(-x)
        }
    }

    /// Chebyshev approximation for erfc
    fn erfccheb(&self, z: f64) -> f64 {
        if z < 0.0 {
            panic!("erfccheb: z must be non-negative");
        }
        let mut d = 0.0;
        let mut dd = 0.0;
        let t = 2.0 / (2.0 + z);
        let ty = 4.0 * t - 2.0;
        for j in (1..Self::NCOF).rev() {
            let tmp = d;
            d = ty * d - dd + Self::COF[j];
            dd = tmp;
        }
        t * (-z * z + 0.5 * (Self::COF[0] + ty * d) - dd).exp()
    }

    /// Inverse complementary error function
    pub fn inverfc(&self, p: f64) -> f64 {
        if p >= 2.0 {
            return -100.0;
        }
        if p <= 0.0 {
            return 100.0;
        }
        let pp = if p < 1.0 { p } else { 2.0 - p };
        let t = (-2.0 * (pp / 2.0).ln()).sqrt();
        let mut x = -f64::consts::FRAC_1_SQRT_2
            * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t);

        for _j in 0..2 {
            let err = self.erfc(x) - pp;
            x += err / (f64::consts::FRAC_2_SQRT_PI * (-x * x).exp() - x * err);
        }

        if p < 1.0 { x } else { -x }
    }

    /// Inverse error function
    pub fn inverf(&self, p: f64) -> f64 {
        self.inverfc(1.0 - p)
    }

    /// Error function complement (alternative implementation)
    pub fn erfcc(&self, x: f64) -> f64 {
        let z = x.abs();
        let t = 2.0 / (2.0 + z);
        let ans = t
            * (-z * z - 1.26551223
                + t * (1.00002368
                    + t * (0.37409196
                        + t * (0.09678418
                            + t * (-0.18628806
                                + t * (0.27886807
                                    + t * (-1.13520398
                                        + t * (1.48851587
                                            + t * (-0.82215223 + t * 0.17087277)))))))))
                .exp();
        if x >= 0.0 { ans } else { 2.0 - ans }
    }
}

impl Default for Erf {
    fn default() -> Self {
        Self::new()
    }
}

/// Normal distribution
pub struct Normaldist {
    mu: f64,
    sig: f64,
    erf: Erf,
}

impl Normaldist {
    /// Create a new Normal distribution
    ///
    /// # Errors
    ///
    /// Returns an error if sigma is not positive.
    pub fn new(mmu: f64, ssig: f64) -> Result<Self, Error> {
        if ssig <= 0.0 {
            bail!(
                "Normal distribution error: sigma must be positive, got {}",
                ssig
            )
        }
        Ok(Self {
            mu: mmu,
            sig: ssig,
            erf: Erf::new(),
        })
    }

    /// Create a new Normal distribution (panics on error)
    ///
    /// # Panics
    ///
    /// Panics if sigma is not positive.
    pub fn new_or_panic(mmu: f64, ssig: f64) -> Self {
        Self::new(mmu, ssig).expect("Failed to create Normal distribution")
    }

    /// Probability density function
    pub fn pdf(&self, x: f64) -> f64 {
        let mu = self.mu;
        let sig = self.sig;
        let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        inv_sqrt_2pi / sig * (-0.5 * ((x - mu) / sig).powi(2)).exp()
    }

    /// Cumulative distribution function
    pub fn cdf(&self, x: f64) -> f64 {
        let mu = self.mu;
        let sig = self.sig;
        0.5 * self.erf.erfc(-f64::consts::FRAC_1_SQRT_2 * (x - mu) / sig)
    }

    /// Inverse cumulative distribution function
    ///
    /// # Errors
    ///
    /// Returns an error if p is not in (0, 1).
    pub fn invcdf(&self, p: f64) -> Result<f64, Error> {
        if p <= 0.0 || p >= 1.0 {
            bail!(
                "Normal distribution error: probability must be in (0, 1), got {}",
                p
            )
        }
        let mu = self.mu;
        let sig = self.sig;
        Ok(-f64::consts::SQRT_2 * sig * self.erf.inverfc(2.0 * p) + mu)
    }
}

impl Default for Normaldist {
    fn default() -> Self {
        Self::new_or_panic(0.0, 1.0)
    }
}

/// Normality critical value
pub fn normality_critical_val() -> f64 {
    Chisqdist::new(2.0)
        .expect("Failed to create Chi-square distribution")
        .invcdf(0.95)
        .expect("Failed to compute inverse CDF")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gammln() {
        // Test some known values
        // Gamma(1) = 1, ln(Gamma(1)) = 0
        assert!((gammln(1.0).expect("gammln(1.0) failed") - 0.0).abs() < 1e-10);
        // Gamma(2) = 1, ln(Gamma(2)) = 0
        assert!((gammln(2.0).expect("gammln(2.0) failed") - 0.0).abs() < 1e-10);
        // Gamma(3) = 2, ln(Gamma(3)) = ln(2)
        assert!((gammln(3.0).expect("gammln(3.0) failed") - 2.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_gammp() {
        // gammp(1, 0) = 0
        assert!((gammp(1.0, 0.0).expect("gammp(1, 0) failed") - 0.0).abs() < 1e-10);
        // gammp(1, inf) = 1 (approximately)
        assert!((gammp(1.0, 100.0).expect("gammp(1, 100) failed") - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_chisqdist() {
        let chi2 = Chisqdist::new(2.0).expect("Failed to create Chisqdist");
        // Chi-square distribution critical value
        let crit = chi2.invcdf(0.95).expect("invcdf failed");
        assert!(crit > 0.0);
        assert!(crit < 10.0);

        // CDF should be monotonically increasing
        assert!(chi2.cdf(1.0).expect("cdf failed") < chi2.cdf(2.0).expect("cdf failed"));
    }

    #[test]
    fn test_fdist() {
        let f = Fdist::new(2.0, 2.0).expect("Failed to create Fdist");
        // F distribution CDF should be in [0, 1]
        assert!(f.cdf(1.0).expect("cdf failed") > 0.0);
        assert!(f.cdf(1.0).expect("cdf failed") < 1.0);
    }

    #[test]
    fn test_gammq() {
        // gammq(1, 0) = 1
        assert!((gammq(1.0, 0.0).expect("gammq(1, 0) failed") - 1.0).abs() < 1e-10);
        // gammq(1, inf) = 0 (approximately)
        assert!(gammq(1.0, 100.0).expect("gammq(1, 100) failed") < 0.01);
        // gammp + gammq = 1
        let a = 2.0;
        let x = 1.5;
        assert!(
            (gammp(a, x).expect("gammp failed") + gammq(a, x).expect("gammq failed") - 1.0).abs()
                < 1e-10
        );
    }

    #[test]
    fn test_erf() {
        let erf = Erf::new();
        // erf(0) = 0
        assert!((erf.erf(0.0) - 0.0).abs() < 1e-10);
        // erf(inf) ≈ 1
        assert!((erf.erf(5.0) - 1.0).abs() < 0.01);
        // erf(-x) = -erf(x)
        assert!((erf.erf(-1.0) + erf.erf(1.0)).abs() < 1e-10);
        // erfc + erf = 1
        let x = 1.5;
        assert!((erf.erf(x) + erf.erfc(x) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normaldist() {
        let norm = Normaldist::new(0.0, 1.0).expect("Failed to create Normaldist");
        // Standard normal distribution CDF(0) = 0.5
        assert!((norm.cdf(0.0) - 0.5).abs() < 1e-10);
        // PDF should be non-negative
        assert!(norm.pdf(0.0) > 0.0);
        assert!(norm.pdf(1.0) > 0.0);
        // CDF should be monotonically increasing
        assert!(norm.cdf(0.0) < norm.cdf(1.0));
        // invcdf(0.5) ≈ 0
        assert!((norm.invcdf(0.5).expect("invcdf failed")).abs() < 0.1);
    }

    #[test]
    fn test_normality_critical_val() {
        let crit = normality_critical_val();
        assert!(crit > 0.0);
        assert!(crit < 10.0);
    }
}

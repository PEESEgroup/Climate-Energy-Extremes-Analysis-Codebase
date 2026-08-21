"""DISC model (Maxwell 1987): GHI -> DNI. Faithful to the standard / pvlib implementation.
disc(ghi, sza_deg, doy, pressure=1013.25 mb) -> DNI (W/m2)."""
import numpy as np
def disc(ghi, sza_deg, doy, pressure=1013.25):
    ghi = np.asarray(ghi, float); z = np.asarray(sza_deg, float); doy = np.asarray(doy, float)
    pressure = np.asarray(pressure, float)
    cosz = np.cos(np.radians(np.clip(z, 0, 90)))
    I0 = 1370.0*(1.0 + 0.033*np.cos(np.radians(360.0*doy/365.0)))      # extraterrestrial normal
    day = (z < 87.0) & (ghi > 0) & (cosz > 0)
    Kt = np.where(day, ghi/np.maximum(I0*cosz, 1e-6), 0.0); Kt = np.clip(Kt, 0, 1)
    # Kasten-Young relative airmass, pressure-corrected, capped
    am = 1.0/np.maximum(cosz + 0.15*np.power(np.maximum(93.885 - z, 1e-3), -1.253), 1e-6)
    am = np.clip(am*(pressure/1013.25), 0, 12.0)
    lo = Kt <= 0.6
    a = np.where(lo, 0.512 - 1.560*Kt + 2.286*Kt**2 - 2.222*Kt**3,
                     -5.743 + 21.77*Kt - 27.49*Kt**2 + 11.56*Kt**3)
    b = np.where(lo, 0.370 + 0.962*Kt,
                     41.40 - 118.5*Kt + 66.05*Kt**2 + 31.90*Kt**3)
    c = np.where(lo, -0.280 + 0.932*Kt - 2.048*Kt**2,
                     -47.01 + 184.2*Kt - 222.0*Kt**2 + 73.81*Kt**3)
    dKn = a + b*np.exp(c*am)
    Knc = 0.866 - 0.122*am + 0.0121*am**2 - 0.000653*am**3 + 0.000014*am**4
    Kn = np.clip(Knc - dKn, 0, None)
    dni = np.where(day, Kn*I0, 0.0)
    return np.clip(dni, 0, 1400.0)

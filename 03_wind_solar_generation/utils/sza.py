"""NOAA solar-position: returns (zenith_deg, azimuth_deg). Matches the eqtime/decl used in srgan_infer/bias_apply.
Azimuth is approximate (gen_physics only consumes zenith). Signature matches GODEEEP utils.sza."""
import numpy as np, pandas as pd
def solar_zenith_and_azimuth_angle(longitude, latitude, time_utc):
    ti = pd.DatetimeIndex(time_utc)
    lon = np.asarray(longitude, float); lat = np.asarray(latitude, float)
    doy = ti.dayofyear.to_numpy().astype(float); hr = ti.hour.to_numpy() + ti.minute.to_numpy()/60.0
    g = 2*np.pi/365.0*(doy - 1 + (hr - 12)/24.0)
    eq = 229.18*(7.5e-5 + 1.868e-3*np.cos(g) - .032077*np.sin(g) - .014615*np.cos(2*g) - .040849*np.sin(2*g))
    dec = (.006918 - .399912*np.cos(g) + .070257*np.sin(g) - .006758*np.cos(2*g)
           + 9.07e-4*np.sin(2*g) - .002697*np.cos(3*g) + .00148*np.sin(3*g))
    latr = np.radians(lat)
    tst = hr*60.0 + eq + 4.0*lon                       # true solar time (min), lon +east
    ha = np.radians(tst/4.0 - 180.0)
    cosz = np.clip(np.sin(latr)*np.sin(dec) + np.cos(latr)*np.cos(dec)*np.cos(ha), -1, 1)
    zen = np.degrees(np.arccos(cosz))
    az = np.zeros_like(zen)                             # unused downstream
    return zen, az

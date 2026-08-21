"""Physics-direct generation: PySAM PVWatts v5 (solar) + Windpower (wind),
matching GODEEEP configs. The DISC irradiance decomposition and the solar-position
routine in utils/ are reimplementations that match the GODEEEP interface.
Weather-at-plant -> capacity factor."""
import sys, numpy as np, pandas as pd
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # local utils/disc.py, utils/sza.py
from utils.disc import disc
from utils.sza import solar_zenith_and_azimuth_angle as sza_saa
import PySAM.Pvwattsv5 as PV
import PySAM.Windpower as WP

def solar_cf(times_utc, ghi, tdry_C, wspd, pres_mb, lat, lon, cfg, elev=0.0, tz=0.0):
    """times_utc: DatetimeIndex; arrays len T. cfg: dict w/ tilt,azimuth,system_capacity,
    module_type,array_type,losses. Returns cf array (relative to DC nameplate)."""
    ti = pd.DatetimeIndex(times_utc)
    sza, _ = sza_saa(longitude=np.full(len(ti), lon), latitude=np.full(len(ti), lat), time_utc=ti)
    dni = disc(np.asarray(ghi,float), np.asarray(sza,float), ti.day_of_year.to_numpy(), pressure=np.asarray(pres_mb,float))
    dhi = np.clip(np.asarray(ghi,float) - dni*np.cos(np.radians(sza)), 0, None)
    m = PV.new()
    m.SolarResource.solar_resource_data = {
        'lat':float(lat),'lon':float(lon),'tz':float(tz),'elev':float(elev),
        'year':ti.year.to_numpy().tolist(),'month':ti.month.to_numpy().tolist(),
        'day':ti.day.to_numpy().tolist(),'hour':ti.hour.to_numpy().tolist(),
        'minute':ti.minute.to_numpy().tolist(),
        'gh':np.asarray(ghi,float).tolist(),'dn':np.nan_to_num(dni).tolist(),'df':dhi.tolist(),
        'tdry':np.asarray(tdry_C,float).tolist(),'wspd':np.asarray(wspd,float).tolist(),
        'pres':np.asarray(pres_mb,float).tolist()}
    sd=m.SystemDesign
    sd.system_capacity=float(cfg.get('system_capacity',1000.0))
    sd.dc_ac_ratio=1.3; sd.inv_eff=96.0; sd.gcr=0.4
    sd.losses=float(cfg.get('losses',14.08)); sd.array_type=int(cfg.get('array_type',0))
    sd.module_type=int(cfg.get('module_type',0)); sd.azimuth=float(cfg.get('azimuth',180.0))
    sd.tilt=float(cfg.get('tilt',abs(lat)))
    m.AdjustmentFactors.adjust_constant=0.0
    m.execute(0)
    ac=np.array(m.Outputs.ac)/1000.0                      # W->kW (AC)
    return ac/sd.system_capacity                          # cf rel. to DC nameplate (kW)

def wind_cf(times_utc, wspd10, tdry_C, pres_mb, lat, lon, cfg, elev=0.0, tz=0.0):
    """cfg: dict w/ wind_turbine_hub_ht, wind_turbine_powercurve_windspeeds/powerout (lists),
    wind_turbine_rotor_diameter, system_capacity. 10m wind extrapolated via shear=0.14."""
    ti=pd.DatetimeIndex(times_utc)
    hub=float(cfg.get('wind_turbine_hub_ht',86.0))
    vhub=np.asarray(wspd10,float)*(hub/10.0)**float(cfg.get("shear",0.14))        # power-law shear 0.14
    m=WP.new()
    m.Resource.wind_resource_model_choice=0
    # single measurement height = hub; columns [temp(C),pres(atm),speed,dir]
    m.Resource.wind_resource_data={'heights':[hub,hub,hub,hub],'fields':[1,2,3,4],
        'data':np.column_stack([np.asarray(tdry_C,float),
                                np.asarray(pres_mb,float)/1013.25,
                                vhub, np.zeros(len(ti))]).tolist()}
    t=m.Turbine
    t.wind_resource_shear=float(cfg.get("shear",0.14))
    t.wind_turbine_hub_ht=hub
    t.wind_turbine_rotor_diameter=float(cfg.get('wind_turbine_rotor_diameter',hub*1.15))
    t.wind_turbine_powercurve_windspeeds=list(cfg['wind_turbine_powercurve_windspeeds'])
    t.wind_turbine_powercurve_powerout=list(cfg['wind_turbine_powercurve_powerout'])
    f=m.Farm
    f.wind_resource_turbulence_coeff=0.1
    f.system_capacity=float(cfg.get('system_capacity',max(cfg['wind_turbine_powercurve_powerout'])))
    f.wind_farm_xCoordinates=[0.0]; f.wind_farm_yCoordinates=[0.0]
    f.wind_farm_wake_model=int(cfg.get('wake_model',0))
    try: m.Losses.turb_generic_loss=float(cfg.get('turb_generic_loss',0.0))   # config specifies 15% (NREL/GODEEEP std); default 0.0 = GROSS (backward-compat)
    except Exception: pass
    m.AdjustmentFactors.adjust_constant=float(cfg.get('loss_pct',cfg.get('adjust_constant',0.0)))  # extra constant haircut if any; default 0
    m.execute(0)
    gen=np.array(m.Outputs.gen)                            # kW
    return gen/f.system_capacity

if __name__=="__main__":   # smoke test: full 8760h year (PVWatts needs multiple of 8760)
    t=pd.date_range("2015-01-01 00:00",periods=8760,freq="h",tz="UTC")
    h=t.hour.to_numpy(); d=t.dayofyear.to_numpy()
    diurnal=np.clip(np.sin((h-6)/12*np.pi),0,None); seasonal=0.7+0.3*np.cos((d-172)/365*2*np.pi)
    ghi=900*diurnal*seasonal
    scfg=dict(system_capacity=5000.0,losses=14.08,array_type=0,module_type=0,azimuth=180.0,tilt=25.0)
    scf=solar_cf(t,ghi,np.full(8760,25.0),np.full(8760,3.0),np.full(8760,1000.0),35.0,-105.0,scfg)
    print("SOLAR: max %.3f mean %.3f  (expect max~0.7-1.0, mean~0.15-0.25)"%(scf.max(),scf.mean()))
    print("  July-15 noon-ish cf:",np.round(scf[(d==196)][10:16],3))
    wcfg=dict(wind_turbine_hub_ht=86.0,wind_turbine_rotor_diameter=113.0,system_capacity=2300.0,
        wind_turbine_powercurve_windspeeds=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
        wind_turbine_powercurve_powerout=[66,171,352,623,1002,1497,2005,2246,2296,2300,2300,2300,2300,2300,2300,2300,2300,2300,2300,2300,2300,2300,2300])
    wspd=6+4*np.sin(np.arange(8760)/50.0)+np.random.default_rng(0).normal(0,1.5,8760)
    wspd=np.clip(wspd,0,None)
    wcf=wind_cf(t,wspd,np.full(8760,20.0),np.full(8760,1000.0),40.0,-100.0,wcfg)
    print("WIND: max %.3f mean %.3f  (expect max~1.0, mean~0.3-0.5)"%(wcf.max(),wcf.mean()))

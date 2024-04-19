import numpy as np
from sedpy.observate import load_filters
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.models.templates import TemplateLibrary
from prospect.sources import FastStepBasis,CSPSpecBasis
from prospect.utils.obsutils import fix_obs
from astropy.io import fits
from prospect.models import SpecModel


Photometry=np.genfromtxt('./photometry.csv',delimiter=',',names=True,missing_values=np.nan)
targets={'NGC3705': np.asarray(list(Photometry[3])[1:]),
         'MCG05': np.asarray(list(Photometry[0])[1:]),
         'MCG06': np.asarray(list(Photometry[1])[1:]),
         'UGC9379': np.asarray(list(Photometry[6])[1:]),
         'NVSSJ09': np.asarray(list(Photometry[5])[1:]),
         'NGC6365A': np.asarray(list(Photometry[4])[1:]),
         'NGC3016': np.asarray(list(Photometry[2])[1:])}
filternames=['galex_FUV','galex_NUV','sdss_u0','Johnson_B','Johnson_V',\
    'Cousins_R','sdss_i0','sdss_z0','twomass_J','twomass_H','twomass_Ks','wise_w1','wise_w2',\
        'wise_w3','wise_w4','AKARI_FIS_N60','AKARI_FIS_WIDE-S','AKARI_FIS_WIDE-L','AKARI_FIS_N160']
spec_scaling={'MCG05':0.2,
'MCG06':0.6,
'NGC3016':0.4,
'NGC3705':None,
'NGC6365A':0.6,
'UGC9379':0.8,
'NVSSJ09':0.3}
redshifts={'MCG05':0.003,
'MCG06':0.02511,
'NGC3016':0.02985,
'NGC3705':0.003,
'NGC6365A':0.02832,
'UGC9379':0.0252,
'NVSSJ09':0.02779}

# run_params = {'verbose': True,
#               'debug': False,
#               'outfile': 'output/demo_mock',
#               'output_pickles': False,
#               # Optimization parameters
#               'do_powell': False,
#               'ftol': 0.5e-5, 'maxfev': 5000,
#               'do_levenberg': True,
#               'nmin': 10,
#               # emcee Fitter parameters
#               'nwalkers': 64,
#               'nburn': [32, 32, 64],
#               'niter': 256,
#               'interval': 0.25,
#               'initial_disp': 0.1,
#               # dynesty Fitter parameters
#               'nested_bound': 'multi',  # bounding method
#               'nested_sample': 'unif',  # sampling method
#               'nested_nlive_init': 100,
#               'nested_nlive_batch': 100,
#               'nested_bootstrap': 0,
#               'nested_dlogz_init': 0.05,
#               'nested_weight_kwargs': {"pfrac": 1.0},
#               'nested_target_n_effective': 10000,
#               # Mock data parameters
#               'snr': 20.0,
#               'add_noise': False,
#               'filterset': galex + sdss + twomass,
#               # Input mock model parameters
#               'mass': 1e10,
#               'logzsol': -0.5,
#               'tage': 12.,
#               'tau': 3.,
#               'dust2': 0.3,
#               'zred': ,
#               'add_neb': True,
#               # SPS parameters
#               'zcontinuous': 1,
#               }

def Build_obs(target,mask_wise,ignore_spec):
    obs={}
    obs['redshift']=redshifts[target]
    obs['filters']=load_filters(filternames)
    obs['maggies']=np.asarray(list(10**(-0.4*targets[target][::2])))
    obs['maggies_unc']=0.921*targets[target][1::2]*obs['maggies']
    if mask_wise==True:
        obs['phot_mask']=[True,True,True,True,True,True,True,True,True,True,True,False,False,True,True,True,True,True,True]
    i=0
    while i<len(obs['maggies']):
        if np.isnan(obs['maggies'][i]):
            del obs['filters'][i]
            if mask_wise:
                del obs['phot_mask'][i]
            obs['maggies']=np.delete(obs['maggies'],i)
            obs['maggies_unc']=np.delete(obs['maggies_unc'],i)
            i+=1
        i+=1
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])
    if target!='NGC3705' and  not ignore_spec :
        spec=fits.open(f'./SDSS_Spec/spec_{target}.fits')[1].data
        obs['wavelength']=10**spec['loglam']
        obs['spectrum']=(spec_scaling[target])*(10**(-7))*spec['flux']/3631
        obs['unc']=(spec_scaling[target])*10**(-7)/(np.sqrt(spec['ivar'])*3631)
        obs['mask']=spec['and_mask']
        obs['rescale_spectrum']=False
    else:
        obs['wavelength']=None
        #obs['spectrum']=None

    obs=fix_obs(obs)
    return obs

def build_model(model_name,obs,fix_tage=False):
    model_params = TemplateLibrary[model_name]
    model_params.update(TemplateLibrary["nebular"])
    model_params["zred"]["init"] = obs["redshift"]
    if fix_tage:
        model_params["tage"]["isfree"]=False
        model_params["tage"]["init"]=12.8
    model = SpecModel(model_params)
    return model

def build_sps(model_name,zcontinuous=1):
    if model_name=="alpha":
        sps = FastStepBasis(zcontinuous=1)
    else:
        sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=False)
    return sps

def build_noise():
    return None, None

def build_all(target,mask_wise,model_name,ignore_spec,fix_tage=False):
    obs=Build_obs(target,mask_wise,ignore_spec)
    model=build_model(model_name,obs,fix_tage)
    sps=build_sps(model_name)
    noise=build_noise()
    return obs,model,sps,noise

if __name__ == '__main__':
    #In serial since was not great otherwise
    obs, model, sps, noise = build_all('NVSSJ09',True,'parametric_sfh',True,False)
    fitting_kwargs = dict(optimize=True,emcee=True,dynesty=False,nwalkers=128,niter=1024)
    output = fit_model(obs, model, sps, **fitting_kwargs)
    result, duration = output["sampling"]
    hfile = "./quickstart_emcee_NVSSJ09.h5"
    writer.write_hdf5(hfile, {}, model, obs,
                 output["sampling"][0], None,
                 sps=sps,
                 tsample=output["sampling"][1],
                 toptimize=0.0)
    obs, model, sps, noise = build_all('NGC6365A',True,'parametric_sfh',True,False)
    fitting_kwargs = dict(optimize=True,emcee=True,dynesty=False,nwalkers=128,niter=1024)
    output = fit_model(obs, model, sps, **fitting_kwargs)
    result, duration = output["sampling"]
    hfile = "./quickstart_emcee_NGC6365A.h5"
    writer.write_hdf5(hfile, {}, model, obs,
                 output["sampling"][0], None,
                 sps=sps,
                 tsample=output["sampling"][1],
                 toptimize=0.0)
    obs, model, sps, noise = build_all('UGC9379',True,'parametric_sfh',True,True)
    fitting_kwargs = dict(optimize=True,emcee=True,dynesty=False,nwalkers=128,niter=1024)
    output = fit_model(obs, model, sps, **fitting_kwargs)
    result, duration = output["sampling"]
    hfile = "./quickstart_emcee_UGC9379.h5"
    writer.write_hdf5(hfile, {}, model, obs,
                 output["sampling"][0], None,
                 sps=sps,
                 tsample=output["sampling"][1],
                 toptimize=0.0)
#Making this since the kernel constantly crashes on the isophotes res
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
import sep
from matplotlib.patches import Ellipse
from astropy.stats import sigma_clipped_stats
from photutils.isophote import EllipseGeometry,Ellipse,build_ellipse_model
import matplotlib as mpl
import scipy.ndimage as im
import json

galaxies={'NGC3705': SkyCoord("11:30:07 +09:16:36",unit=('hourangle,deg')),
         'MCG05': SkyCoord("17:06:55 +30:16:11",unit=('hourangle,deg')),
         'MCG06': SkyCoord("12:12:05 +32:44:06",unit=('hourangle,deg')),
         'UGC9379': SkyCoord("14:33:59 +40:14:40",unit=('hourangle,deg')),
         'NVSSJ09': SkyCoord("09:24:57 +40:23:59",unit=('hourangle,deg')),
         'NGC6365A': SkyCoord("17:22:44 +62:09:58",unit=('hourangle,deg')),
         'NGC3016': SkyCoord("09:49:51 +12:41:43",unit=('hourangle,deg'))}
SN_pos={'NGC3705': SkyCoord("11:30:05.940 +09:16:57.37",unit=('hourangle,deg')),
         'MCG05': SkyCoord("17:06:54.600 +30:16:17.40",unit=('hourangle,deg')),
         'MCG06': SkyCoord("12:12:04.895 +32:44:01.73",unit=('hourangle,deg')),
         'UGC9379': SkyCoord("14:33:57.009 +40:14:37.62",unit=('hourangle,deg')),
         'NVSSJ09': SkyCoord("09:24:57.173 +40:23:55.14",unit=('hourangle,deg')),
         'NGC6365A': SkyCoord("17:22:44.400 +62:09:44.17",unit=('hourangle,deg')),
         'NGC3016': SkyCoord("09:49:50.515 +12:41:44.20",unit=('hourangle,deg'))}
full_names={'NGC3705': 'NGC 3705',
    'MCG05': 'MCG+05-40-038',
    'MCG06': 'MCG+06-27-025',
    'NGC3016': 'NGC 3016',
    'NGC6365A': 'NGC 6365A',
    'UGC9379': 'UGC 9379',
    'NVSSJ09': 'NVSS J092456+402359'}
sn_names={'NGC3705':'SN 2022xxf',
    'MCG05': 'SN 2020lao',
    'MCG06': 'SN 2020ayz',
    'NGC3016': 'SN 2019eto',
    'NGC6365A': 'SN 2016ino',
    'UGC9379': 'SN 2020bvc',
    'NVSSJ09': 'SN 2018kq'}

def Coord_Rotator(x,y,pa):
    x_rot=x*np.cos(pa*(np.pi/180))-y*np.sin(pa*(np.pi/180))
    y_rot=x*np.sin(pa*(np.pi/180))+y*np.cos(pa*(np.pi/180))
    return x_rot,y_rot
def Padded_Coord_Rotator(x,y,pa,padX,padY):
    x_new=x-padX[1]
    y_new=y-padY[1]
    x_rot=x_new*np.cos(pa*(np.pi/180))-y_new*np.sin(pa*(np.pi/180))
    y_rot=x_new*np.sin(pa*(np.pi/180))+y_new*np.cos(pa*(np.pi/180))
    x_rot+=padX[0]+padX[1]
    y_rot+=padY[0]+padY[1]
    return x_rot,y_rot

gals=list(galaxies.keys())
b_a_dict={}
cosi_dict={}
offset_dict={}
r50_dict={}
r90_dict={}
sma_dict={}
PA_dict={}
fig, axes = plt.subplots(7, 3,figsize=(15,35),layout='constrained')
for num,gal in enumerate(gals):
    im_test=f'./WCS_Solved/{gal}/{gal}_R.fits'
    image=fits.open(im_test)[0].data
    hdr=fits.open(im_test)[0].header
    wcs=WCS(hdr)
    data=image.byteswap().newbyteorder()
    gal_x_init,gal_y_init=wcs.world_to_pixel(galaxies[gal])
    sn_x,sn_y=wcs.world_to_pixel(SN_pos[gal])
    bkg = sep.Background(data)
    data_sub=data-bkg.globalback
    if gal=='NGC6365A':
        objects,seg_map = sep.extract(data_sub, 3, err=bkg.globalrms,segmentation_map=True,deblend_cont=.005)
    else:
        objects,seg_map = sep.extract(data_sub, 3, err=bkg.globalrms,segmentation_map=True,deblend_cont=1)#use if no blending wanted
    dists=np.sqrt((objects['x']-gal_x_init)**2+(objects['y']-gal_y_init)**2)
    obj=np.argmin(dists)
    x=objects['x'][obj]
    y=objects['y'][obj]
    a=objects['a'][obj]
    b=objects['b'][obj]
    theta=objects['theta'][obj]
    obj_seg=seg_map[int(y),int(x)]
    bkg_mask=np.zeros(data.shape,dtype=bool)
    masked_pixels=0
    for i in range(len(data)):
        for j in range(len(data[i])):
            if seg_map[i][j]!=0:
                bkg_mask[i][j]=True
                masked_pixels+=1
    newbkg=sep.Background(data,mask=bkg_mask)
    bkg_mean,bkg_med,bkg_std=sigma_clipped_stats(newbkg,sigma=2)
    test=image.copy()
    for i in range(len(test)):
        for j in range(len(test[i])):
            #if seg_map[i][j]!=0:
            if (seg_map[i][j]!=obj_seg) and seg_map[i][j]!=0: 
            #if (seg_map[i][j]!=obj_seg) and (seg_map[i][j]!=obj_seg+1)and seg_map[i][j]!=0: for NGC3705, replace previous
                test[i][j]=bkg_med
    iso_data=test-bkg_med
    geo=EllipseGeometry(x,y,a,np.sqrt(1-(b/a)**2),theta)
    ellipse = Ellipse(iso_data, geo)
    isos = ellipse.fit_image()
    model_image = build_ellipse_model(iso_data.shape, isos)
    residual = iso_data - model_image
    tol=int(1.1*np.max(isos.sma))
    ax1=axes[num,0]
    ax2=axes[num,1]
    ax3=axes[num,2]
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax1.imshow(iso_data,vmin=0,vmax=500,cmap='inferno')
    ax1.set_ylim(int(y)-tol,int(y)+tol)
    ax1.set_xlim(int(x)-tol,int(x)+tol)
    smas = np.linspace(np.min(isos.sma), np.max(isos.sma), 20)
    half_flux=0.5*np.nanmax(isos.tflux_e)
    iso_ind_50=np.nanargmin([np.abs(el-half_flux) for el in isos.tflux_e])
    sma_r50=isos.sma[iso_ind_50]
    iso_50=isos.get_closest(sma_r50)
    x2,y2=iso_50.sampled_coordinates()
    flux90=0.9*np.nanmax(isos.tflux_e)
    iso_ind_90=np.nanargmin([np.abs(el-flux90) for el in isos.tflux_e])
    sma_r90=isos.sma[iso_ind_90]
    iso_90=isos.get_closest(sma_r90)
    x3,y3=iso_90.sampled_coordinates()
    #iso=isos
    ax2.imshow(model_image,vmin=0,vmax=500,cmap='inferno')
    ax2.set_ylim(int(y)-tol,int(y)+tol)
    ax2.set_xlim(int(x)-tol,int(x)+tol)
    res_tol=10*sigma_clipped_stats(residual)[2]
    ax3.imshow(residual,vmin=-res_tol,vmax=res_tol,cmap='inferno')
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=-res_tol, vmax=res_tol)
    ax3.set_ylim(int(y)-tol,int(y)+tol)
    ax3.set_xlim(int(x)-tol,int(x)+tol)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax3)
    pix_max=np.unravel_index(np.argmax(iso_data),iso_data.shape)
    ax1.set_title(f'{full_names[gal]} Image')
    ax2.set_title(f'{full_names[gal]} Isophote Model')
    ax3.set_title(f'{full_names[gal]} Residuals and Colourbar')
    pa=90-isos.pa[iso_ind_90]*(180/np.pi)
    padX = [iso_data.shape[1] - round(x), round(x)]
    padY = [iso_data.shape[0] - round(y), round(y)]
    sn_x_adj,sn_y_adj=Padded_Coord_Rotator(sn_x,sn_y,pa,padX,padY)
    gal_x_med,gal_y_med=Padded_Coord_Rotator(np.median(isos.x0),np.median(isos.y0),pa,padX,padY)
    cos_i=(((1-isos.eps[iso_ind_90])**2-(0.2**2)))/(1-0.2**2)
    b_a=1-isos.eps[iso_ind_90]
    offset=np.sqrt((sn_x_adj-gal_x_med)**2+((sn_y_adj-gal_y_med)/cos_i)**2)*0.6
    R_50=(sma_r50/cos_i)*np.sqrt((1-isos.eps[iso_ind_50]))*0.6
    R_90=(sma_r90/cos_i)*np.sqrt((1-isos.eps[iso_ind_90]))*0.6
    b_a_dict[gal]=b_a
    cosi_dict[gal]=cos_i
    offset_dict[gal]=offset
    r50_dict[gal]=R_50
    r90_dict[gal]=R_90
    sma_dict[gal]=sma_r50/cos_i
    PA_dict[gal]=pa
    ax1.plot(x2,y2,color='r',zorder=10,label=f'$r_{{50}}={R_50:.0f}$\"')
    ax1.plot(x3,y3,color='r',zorder=10,ls='--',label=f'$r_{{90}}={R_90:.0f}$\"')
    ax1.scatter(sn_x,sn_y,[100],marker='*',color='red',label=sn_names[gal])
    ax1.scatter(pix_max[1],pix_max[0],[30],marker='+',color='green',label='Brightest Pixel')
    ax1.scatter(np.median(isos.x0),np.median(isos.y0),[30],marker='x',color='red',label='Median Centroid')
    ax1.legend(loc='lower left')
plt.savefig('./Isophotes_final3.png')
#This is a different way but going to run with it
result_dict={}
for gal in galaxies.keys():
    result_dict[gal]={'b_a':b_a_dict[gal],'cos_i':cosi_dict[gal],'offset':offset_dict[gal],'r_50':r50_dict[gal],'r_90':r90_dict[gal],'sma50':sma_dict[gal],'PA':PA_dict[gal]}
with open('./iso_res.json', 'w', encoding='utf-8') as f:
    json.dump(result_dict,f,ensure_ascii=False, indent=4)
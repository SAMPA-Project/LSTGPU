#pragma once

/* Model constants: the scalar parameters and the material table. The defaults are the values calibrated against
   Landsat 8/9 LST (see README.md, "Model constants"); a text file (--params) can override any of them:

     key value                                                      # comment
     class i albedo emissivity a1 a2 a3 alpha_pt field_capacity_mm  # one row of the material table

   params/default_params.txt lists every constant in this format. */

#include <string>

#include "lst_constants.h"

#define P_HC_SCALE          0   // multiplier of the convective heat transfer coefficient
#define P_HC_MIN            1   // floor of h_c before scaling (W m-2 K-1)
#define P_WIND_MIN          2   // floor of the local wind speed (m s-1)
#define P_WIND_EXP          3   // power-law wind profile exponent
#define P_WIND_ZREF         4   // reference height of the measured wind (m)
#define P_SEAS_AMP          5   // amplitude of the seasonal storage term (W m-2), + in winter
#define P_SEAS_PHASE        6   // day of year of its maximum
#define P_CANOPY_K          7   // Beer-Lambert extinction coefficient for LAI (radiation, canopy fraction)
#define P_WIND_CAN_K        8   // extinction for the canopy wind reduction
#define P_WIND_CAN_RED      9   // maximum wind reduction inside the canopy
#define P_SHELTER_GAMMA    10   // terrain/building wind sheltering exponent
#define P_SM_INIT          11   // initial soil moisture as a fraction of field capacity
#define P_SM_LOSS          12   // daily drainage fraction of soil moisture (soil_water_v2 0 only)
#define P_GAMMA_PT         13   // psychrometric constant in Priestley-Taylor (kPa K-1)
#define P_BRUTSAERT        14   // clear-sky emissivity coefficient (Brutsaert 1975: 1.24)
#define P_LSURR_FB_EPS     15   // emissivity of the fallback Lsurr (cells without sampled sources)
#define P_REFL_FB_ALB      16   // albedo of the fallback reflected shortwave
#define P_VEG_LW_TRANS     17   // 1: radiation transmitted through a canopy in the Lsurr sum comes from surfaces at air temperature; 0: lost
#define P_LSURR_RAYW       18   // 1: Lsurr and reflected shortwave weighted per hemisphere ray (hits x bin solid angle x cos / pi); 0: 1 m2 form factor per voxel
#define P_SOIL_V2          19   // 1: the soil water bucket limits latent heat in the energy balance and is depleted by it, no drainage
                                //    below field capacity, gradual water stress, open water never dries out; 0: simple bucket
#define P_OPEN_WATER       20   // material class of open water (never loses water; soil_water_v2), -1 = none
#define P_ET_STRESS        21   // latent heat reduced linearly below this fraction of field capacity (FAO-56 style; soil_water_v2), 0 = none
#define P_CANOPY_TAU       22   // 1: in the exchange sums (Lsurr, reflected shortwave) a source seen through a canopy is weighted by the
                                //    canopy transmittance tau = exp(-k LAI), as the direct beam, and the canopy (1 - tau) emits at air
                                //    temperature (with veg_lw_transmitted); 0: the source is weighted by the canopy fraction 1 - tau
#define P_WIND_AGL         23   // 1: the wind profile height is the voxel's height above the local ground (street level 0, roofs the
                                //    building height); 0: its height above the lowest ground point of the simulated area
#define P_NUM              24

// columns of a material table row
#define M_ALBEDO     0
#define M_EMISSIVITY 1
#define M_OHM_A1     2
#define M_OHM_A2     3 // h
#define M_OHM_A3     4 // W m-2
#define M_ALPHA_PT   5 // Priestley-Taylor alpha
#define M_FIELD_CAP  6 // field capacity (mm)
#define M_NUM        7

namespace lstgpu {

extern const char* const kParamNames[P_NUM];
extern const char* const kMaterialNames[NUM_MATERIALS];

struct ModelParams {
    float p[P_NUM];
    float materials[NUM_MATERIALS][M_NUM];
};

// The calibrated defaults.
ModelParams default_params();

/* Overrides `mp` from a parameter file. Unknown keys and malformed class rows are reported in `log`.
   Returns false if the file can't be opened. */
bool read_params(const std::string& path, ModelParams& mp, std::string& log);

void write_params(const std::string& path, const ModelParams& mp);

} // namespace lstgpu

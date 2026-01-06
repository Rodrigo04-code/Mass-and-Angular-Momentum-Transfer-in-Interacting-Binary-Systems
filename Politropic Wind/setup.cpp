/*
Author: Rodrigo Pérez San Martín 
Description:
- politropic Parker Wind Initialization for Idefix
- Implements a non-dimensional spherically symmetric wind
- Uses Newton-Raphson to solve the politropic Bernoulli equation for velocity
- Floor values applied to velocity, density, and pressure to avoid numerical divergences
- Custom boundary conditions: inflow at inner boundary, free outflow at outer boundary
- Transonic flow initialized centered at the critical sonic radius rs, covering subsonic and supersonic branches
*/

#include <Kokkos_Core.hpp>
#include "idefix.hpp"
#include "setup.hpp"
#include <cmath>

real GAMMA_GLOB;

KOKKOS_INLINE_FUNCTION real get_v_floor()   { return 1e-12; }
KOKKOS_INLINE_FUNCTION real get_rho_floor() { return 1e-12; }
KOKKOS_INLINE_FUNCTION real get_prs_floor() { return 1e-12; }

KOKKOS_INLINE_FUNCTION real get_cs2_politropic(real v, real r, real rs, real vs, real gam) {
    real rho_ratio = (rs*rs*vs) / (r*r*v);
    return vs*vs * pow(rho_ratio, gam - 1.0);
}

KOKKOS_INLINE_FUNCTION real fParkerpolitropic(real v, real r, real rs, real vs, real gam) {
    real cs2 = get_cs2_politropic(v, r, rs, vs, gam);
    return 0.5*v*v + cs2/(gam-1.0) - (2.0/r) - 0.5;
}

KOKKOS_INLINE_FUNCTION real dfParkerpolitropic(real v, real r, real rs, real vs, real gam) {
    real cs2 = get_cs2_politropic(v, r, rs, vs, gam);
    return v - (cs2 / v);
}

KOKKOS_INLINE_FUNCTION real solve_v_at_point(real r, real rs, real vs, real gam, real v_guess) {
    real v = v_guess;
    for(int it=0; it<50; it++) {
        real f = fParkerpolitropic(v, r, rs, vs, gam);
        real df = dfParkerpolitropic(v, r, rs, vs, gam);
        if(fabs(df) < 1e-12) df = (df > 0 ? 1e-12 : -1e-12);
        real dv = -f / df;
        v += dv;
        if(fabs(dv/v) < 1e-10) break;
        if(v < 1e-12) v = 1e-12; 
    }
    return v;
}

void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
    if(dir != IDIR) return;
    IdefixArray4D<real> Vc = hydro->Vc;
    auto *data = hydro->data;
    DataBlockHost d(*data); 

    real gam = GAMMA_GLOB;
    real v_inf_sq = 1.0;
    real num = v_inf_sq * (gam - 1.0);
    real den = 5.0 - 3.0 * gam;
    if (den <= 0) return; 

    real vs = sqrt(num / den);
    real rs = 2.0 / (2.0 * vs * vs);

    if(side == left) {
        int ibeg = 0;
        int iend = data->beg[IDIR];
        idefix_for("BoundaryLeft", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend,
                   KOKKOS_LAMBDA(int k, int j, int i) {
            real r_ghost = data->x[IDIR](i);
            real v_exact = solve_v_at_point(r_ghost, rs, vs, gam, vs*0.1);
            real rho_exact = 1 / (r_ghost*r_ghost*v_exact);
            real cs2 = get_cs2_politropic(v_exact, r_ghost, rs, vs, gam);
            real prs_exact = rho_exact * cs2 / gam;
            Vc(VX1,k,j,i) = v_exact;
            Vc(RHO,k,j,i) = rho_exact;
            Vc(PRS,k,j,i) = prs_exact;
            Vc(VX2,k,j,i) = 0.0;
            Vc(VX3,k,j,i) = 0.0;
        });
    }
    else if(side == right) {
        int ighost = data->end[IDIR]-1;
        int ibeg = data->end[IDIR];
        int iend = data->np_tot[IDIR];
        idefix_for("BoundaryRight", 0, data->np_tot[KDIR], 0, data->np_tot[JDIR], ibeg, iend,
                   KOKKOS_LAMBDA(int k, int j, int i) {
            Vc(VX1,k,j,i) = Vc(VX1,k,j,ighost);
            Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost);
            Vc(PRS,k,j,i) = Vc(PRS,k,j,ighost);
            Vc(VX2,k,j,i) = 0.0;
            Vc(VX3,k,j,i) = 0.0;
        });
    }
}

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    GAMMA_GLOB = input.Get<real>("Hydro", "gamma",0);
    data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
}

void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);
    int n1 = d.np_tot[IDIR];
    
    real gam = GAMMA_GLOB;
    real v_inf_sq = 1.0;
    real num = v_inf_sq * (gam - 1.0);
    real den = 5.0 - 3.0 * gam;
    if (den <= 0) { idfx::cout << "Error Gamma" << std::endl; exit(1); }
    real vs = sqrt(num / den);
    real rs = 2.0 / (2.0 * vs * vs);

    int ic = 0;
    for(int i=0; i<n1; i++) {
        real r = d.x[IDIR](i);
        if(fabs(r - rs) < fabs(d.x[IDIR](ic) - rs)) ic = i;
    }

    std::vector<real> v_sol(n1);
    v_sol[ic] = vs;

    for(int i = ic; i >= 0; i--) {
        real r = d.x[IDIR](i);
        real guess = (i==ic) ? vs*0.99 : v_sol[i+1];
        v_sol[i] = solve_v_at_point(r, rs, vs, gam, guess);
    }
    for(int i = ic+1; i < n1; i++) {
        real r = d.x[IDIR](i);
        real guess = v_sol[i-1]*1.01;
        v_sol[i] = solve_v_at_point(r, rs, vs, gam, guess);
    }

    for(int k=0; k<d.np_tot[KDIR]; k++) {
        for(int j=0; j<d.np_tot[JDIR]; j++) {
            for(int i=0; i<n1; i++) {
                real r = d.x[IDIR](i);
                d.Vc(VX1,k,j,i) = v_sol[i];
                real rho = 1/ (r*r*v_sol[i]);
                d.Vc(RHO,k,j,i) = rho;
                real cs2 = get_cs2_politropic(v_sol[i], r, rs, vs, gam);
                d.Vc(PRS,k,j,i) = rho * cs2 / gam;
                d.Vc(VX2,k,j,i) = 0.0;
                d.Vc(VX3,k,j,i) = 0.0;
            }
        }
    }
    d.SyncToDevice();
}


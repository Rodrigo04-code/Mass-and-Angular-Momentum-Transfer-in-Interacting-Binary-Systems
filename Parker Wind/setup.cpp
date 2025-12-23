#include <Kokkos_Core.hpp>
#include "idefix.hpp"
#include "setup.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

// Hydrodynamic Parker Wind Setup for Idefix

// Author: Rodrigo Pérez San Martín

// This code implements the initialization of a spherically symmetric Parker wind
// in a non-dimensional formulation. The velocity profile is determined by solving
// the Parker transcendental equation using the Newton-Raphson method. Floor values
// for velocity and density are applied to avoid numerical instabilities. Custom
// boundary conditions are enforced, with an inflow at the inner boundary and a
// free outflow at the outer boundary. The flow is initialized as a transonic solution
// centered at the critical radius r_c = 1.0, ensuring a smooth transition between
// subsonic and supersonic branches of the wind. The resulting profiles are then
// assigned to the computational DataBlock for subsequent hydrodynamic evolution.




// Floor constants

KOKKOS_INLINE_FUNCTION real get_v_floor()   { return 1e-12; }
KOKKOS_INLINE_FUNCTION real get_rho_floor() { return 1e-12; }


// Parker transcendental function (non-dimensional)
// v^2 - ln(v^2) = 4 ln r + 4/r - 3

KOKKOS_INLINE_FUNCTION real fParker(real v, real r) {
    return v*v - log(v*v) - (4.0*log(r) + 4.0/r - 3.0);
}

KOKKOS_INLINE_FUNCTION real dfParker(real v) {
    return 2.0*v - 2.0/v;
}


// Newton-Raphson method to solve v(r)

real solve_v_newton(real r, real v_init = 1.0) {
    const int max_iter = 1000;
    const real tol = 1e-12;
    real v = v_init;

    for(int it=0; it<max_iter; it++) {
        real f = fParker(v, r);
        real df = dfParker(v);
        if(fabs(df) < 1e-12) df = (df > 0 ? 1e-12 : -1e-12);
        real dv = -f / df;
        v += dv;
        if(fabs(dv) < tol) break;
        if(v < 1e-8) v = 1e-8;
    }
    return v;
}


// Boundary Conditions

// This function defines custom boundary conditions along the radial direction:
// - Inner boundary (left): inflow condition, velocity and density are set
//   using the adjacent active cell while ensuring floor values
// - Outer boundary (right): free outflow, values are copied from the last
//   active cell

void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
    if(dir != IDIR) return;

    IdefixArray4D<real> Vc = hydro->Vc;
    auto *data = hydro->data;

    // Inner boundary: inflow
    if(side == left) {
        int ibeg = 0;
        int iend = data->beg[IDIR];
        idefix_for("BoundaryLeft", 0, data->np_tot[KDIR],
                   0, data->np_tot[JDIR], ibeg, iend,
                   KOKKOS_LAMBDA(int k, int j, int i) {
            int src = data->beg[IDIR];
            Vc(VX1,k,j,i) = Vc(VX1,k,j,src);
            Vc(RHO,k,j,i) = fmax(Vc(RHO,k,j,src), get_rho_floor());
            Vc(VX2,k,j,i) = 0.0;
            Vc(VX3,k,j,i) = 0.0;
        });
    }

    // Outer boundary: free outflow
    else if(side == right) {
        int ighost = data->end[IDIR]-1;
        int ibeg = data->end[IDIR];
        int iend = data->np_tot[IDIR];
        idefix_for("BoundaryRight", 0, data->np_tot[KDIR],
                   0, data->np_tot[JDIR], ibeg, iend,
                   KOKKOS_LAMBDA(int k, int j, int i) {
            Vc(VX1,k,j,i) = Vc(VX1,k,j,ighost);
            Vc(RHO,k,j,i) = fmax(Vc(RHO,k,j,ighost), get_rho_floor());
            Vc(VX2,k,j,i) = 0.0;
            Vc(VX3,k,j,i) = 0.0;
        });
    }
}


// Setup Constructor

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
}


// Initialize the Flow (Parker transonic, wind branch)


void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);

    int n1 = d.np_tot[IDIR];
    std::vector<real> v_profile(n1), rho_profile(n1), r_profile(n1);

    // Radial coordinates
    for(int i=0; i<n1; i++) r_profile[i] = d.x[IDIR](i);

    // Find index closest to critical point r_c = 1
    int ic = 0;
    for(int i=0; i<n1; i++)
        if(fabs(r_profile[i]-1.0) < fabs(r_profile[ic]-1.0)) ic = i;

    // Enforce critical point
    v_profile[ic] = 1.0;
    rho_profile[ic] = 1.0 / (r_profile[ic]*r_profile[ic]*v_profile[ic]);

    // Integration inward (i < ic) → subsonic
    for(int i = ic-1; i >= 0; i--) {
        real v_guess = v_profile[i+1] * 0.95; // slightly smaller seed
        v_profile[i] = solve_v_newton(r_profile[i], v_guess);
        v_profile[i] = fmax(v_profile[i], get_v_floor());
        rho_profile[i] = fmax(1.0 / (r_profile[i]*r_profile[i]*v_profile[i]), get_rho_floor());
    }

    // Integration outward (i > ic) → supersonic wind branch
    for(int i = ic+1; i < n1; i++) {
        real v_guess = v_profile[i-1] * 1.05; // slightly larger seed
        v_profile[i] = solve_v_newton(r_profile[i], v_guess);
        v_profile[i] = fmax(v_profile[i], get_v_floor());
        rho_profile[i] = fmax(1.0 / (r_profile[i]*r_profile[i]*v_profile[i]), get_rho_floor());
    }

    // Assign profiles to the DataBlock
    for(int k=0; k<d.np_tot[KDIR]; k++)
        for(int j=0; j<d.np_tot[JDIR]; j++)
            for(int i=0; i<n1; i++) {
                d.Vc(RHO,k,j,i) = rho_profile[i];
                d.Vc(VX1,k,j,i) = v_profile[i];
                d.Vc(VX2,k,j,i) = 0.0;
                d.Vc(VX3,k,j,i) = 0.0;
            }

    d.SyncToDevice();
}


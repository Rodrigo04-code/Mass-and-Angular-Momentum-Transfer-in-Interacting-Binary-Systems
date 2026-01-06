/*
Author: Rodrigo Pérez San Martín
Project: Mass transfer in binary system
Description: This code implements the physical setup required to simulate mass transfer in a binary system using the Idefix hydrodynamic framework. The program defines a user-specified Roche gravitational potential in Cartesian coordinates and computes its representation on a spherical grid. It also enrolls the corresponding body force terms associated with rotation in the corotating frame. A custom internal boundary condition is applied to mimic an accretion sink near the primary object and to maintain an polytropic atmosphere around the donor star close to the L1 point. Initial conditions for density and velocity are generated based on the Roche potential to create a stable configuration prior to the onset of the overflow. The overall objective of the code is to provide a controlled numerical environment for studying disk formation and outflow structures resulting from Roche lobe overflow in close binary systems.
*/
// Note: The "Adiabatic" function is a conceptual error, the equations is for a polytropic RLOF 

#include "idefix.hpp"
#include "setup.hpp"
#include <cmath>
#include <iostream>

// Parameters
real qGlob;
real L1Glob;
real rho_surfGlob;
real densityFloorGlob;
real pot_L1Glob;
real Gamma_Glob; // From .ini


const real soft_eps = 0.01; 
const real soft_eps2 = soft_eps*soft_eps;
const real r1x = 1.0; 
const real r2x = 0.0; 
const real ffactor = 1.1; 

// For initial reference sound speed
const real cs2_ref = 0.4*0.4; 


// P = K * rho^gamma
KOKKOS_INLINE_FUNCTION
real AdiabaticPressure(real rho, real rho_ref, real cs2_ref, real gamma) {
  // P_ref = (rho * cs^2) / gamma
  real P_ref = (rho_ref * cs2_ref) / gamma;
  return P_ref * pow(rho / rho_ref, gamma);
}
// Roche Potential
KOKKOS_INLINE_FUNCTION
real RochePotentialAtXYZ(real X, real Y, real Z, real q, real xcm) {
  real r1 = sqrt((X - r1x)*(X - r1x) + Y*Y + Z*Z + soft_eps2);
  real r2 = sqrt((X - r2x)*(X - r2x) + Y*Y + Z*Z + soft_eps2);
  real r2xy_cm = (X - xcm)*(X - xcm) + Y*Y;
  return - q / r1 - 1.0 / r2 - 0.5 * (1.0 + q) * r2xy_cm;
} 


void Potential(DataBlock& data, const real t,
               IdefixArray1D<real>& x1,
               IdefixArray1D<real>& x2,
               IdefixArray1D<real>& x3,
               IdefixArray3D<real>& phi) {
  real q = qGlob;
  real xcm = q / (1.0 + q);
  idefix_for("Potential", 0, data.np_tot[KDIR], 0, data.np_tot[JDIR], 0, data.np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {
      real r = x1(i);
      real th = x2(j);
      real ph = x3(k);
      real X = r * sin(th) * cos(ph);
      real Y = r * sin(th) * sin(ph);
      real Z = r * cos(th);
      phi(k,j,i) = RochePotentialAtXYZ(X, Y, Z, q, xcm);
    });
}

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
  idfx::pushRegion("BodyForce");
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray1D<real> x3 = data.x[KDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;
  real q = qGlob;

  idefix_for("BodyForce",
              data.beg[KDIR] , data.end[KDIR],
              data.beg[JDIR] , data.end[JDIR],
              data.beg[IDIR] , data.end[IDIR],
              KOKKOS_LAMBDA (int k, int j, int i) {
                real th = x2(j);
                real v3 = Vc(VX3,k,j,i);
                if (!(v3 == v3)) v3 = 0.0;
                force(IDIR,k,j,i) = 2.0*sqrt(1+q)* (v3 * sin(th));
                force(JDIR,k,j,i) = 2.0*sqrt(1+q)* (v3 * cos(th));
                force(KDIR,k,j,i) = -2.0*sqrt(1+q)* (Vc(VX2,k,j,i)*cos(th)+Vc(VX1,k,j,i)*sin(th));
      });
  idfx::popRegion();
}

// Boundary

void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  if (dir != IDIR) return;
  if (side != left) return; 

  auto *data = hydro->data;
  IdefixArray1D<real> x1 = data->x[IDIR];
  IdefixArray1D<real> x2 = data->x[JDIR];
  IdefixArray1D<real> x3 = data->x[KDIR];
  IdefixArray4D<real> Vc = data->hydro->Vc;

  real gamma_local = Gamma_Glob;
  real q_local = qGlob;
  real L1_local = L1Glob;
  real rho_surf = rho_surfGlob;
  real rho_min = densityFloorGlob;
  real xcm = q_local / (1.0 + q_local);
  real phiL1 = pot_L1Glob;
  real R_lobe = fabs(r1x - L1_local);
  real Rcut = ffactor * R_lobe;
  real rmin = 0.06; 


  const real RHO_SAFE_MIN = fmax(rho_min, (real)1e-8);

  const real PRS_SAFE_MIN = AdiabaticPressure(RHO_SAFE_MIN, rho_surf, cs2_ref, gamma_local);
  
  const real V_MAX        = (real)1e3;
  const real RHO_MAX_WARN = (real)1e6;
  const real PRS_MAX_WARN = (real)1e8;
  const real PRS_MIN_ABS  = (real)1e-12;


  idefix_for("UserDefBoundary",
    0, data->np_tot[KDIR],
    0, data->np_tot[JDIR],
    0, data->np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {
      real r = x1(i);
      real th = x2(j);
      real ph = x3(k);
      real X = r * sin(th) * cos(ph);
      real Y = r * sin(th) * sin(ph);
      real Z = r * cos(th);
      real pot = RochePotentialAtXYZ(X, Y, Z, q_local, xcm);
      real dx = X - r1x;
      real dist_M1 = sqrt(dx*dx + Y*Y + Z*Z);

      // Leer estado
      real rho_cell = Vc(RHO,k,j,i);
      real prs_cell = Vc(PRS,k,j,i);
      real vx1 = Vc(VX1,k,j,i);
      real vx2 = Vc(VX2,k,j,i);
      real vx3 = Vc(VX3,k,j,i);

      if (!(rho_cell == rho_cell)) rho_cell = RHO_SAFE_MIN;

      // L_1
      if (dist_M1 < fabs(Rcut) && pot < phiL1) {
        rho_cell = fmax(rho_surf, RHO_SAFE_MIN);
        vx1 = 0.0;
        vx2 = 0.0;
        vx3 = 0.0;
        
        prs_cell = AdiabaticPressure(rho_cell, rho_surf, cs2_ref, gamma_local);
      }

      
      if (rho_cell < rho_min) {
          rho_cell = rho_min;
         
          prs_cell = PRS_SAFE_MIN;
      }

     
      if (r < rmin) {
        const real alpha = (real)0.2; 
        rho_cell = (1.0 - alpha) * rho_cell + alpha * RHO_SAFE_MIN;
        vx1 = (1.0 - alpha) * vx1;
        vx2 = (1.0 - alpha) * vx2;
        vx3 = (1.0 - alpha) * vx3;
        
      
        prs_cell = PRS_SAFE_MIN; 
      }

      // Clamps
      if (fabs(vx1) > V_MAX) vx1 = (vx1 > 0 ? V_MAX : -V_MAX);
      if (fabs(vx2) > V_MAX) vx2 = (vx2 > 0 ? V_MAX : -V_MAX);
      if (fabs(vx3) > V_MAX) vx3 = (vx3 > 0 ? V_MAX : -V_MAX);
      if (rho_cell > RHO_MAX_WARN) rho_cell = fmin(rho_cell, RHO_MAX_WARN);
      if (prs_cell > PRS_MAX_WARN) prs_cell = fmin(prs_cell, PRS_MAX_WARN);
      if (prs_cell < PRS_MIN_ABS) prs_cell = PRS_SAFE_MIN;

      // Reasignar
      Vc(RHO,k,j,i) = rho_cell;
      Vc(PRS,k,j,i) = prs_cell;
      Vc(VX1,k,j,i) = vx1;
      Vc(VX2,k,j,i) = vx2;
      Vc(VX3,k,j,i) = vx3;
    });
}

//  Setup

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  data.gravity->EnrollPotential(&Potential);
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);
  data.gravity->EnrollBodyForce(BodyForce);

  rho_surfGlob     = input.Get<real>("Setup","rho_surf",0);
  qGlob            = input.Get<real>("Setup","q",0);
  L1Glob           = input.Get<real>("Setup","L1",0);
  densityFloorGlob = input.Get<real>("Setup","densityFloor",0);
  Gamma_Glob       = input.Get<real>("Hydro","gamma",0);

  real xcm = qGlob / (1.0 + qGlob);
  pot_L1Glob = RochePotentialAtXYZ(L1Glob, 0.0, 0.0, qGlob, xcm);
  

}
// Initial Conditions
void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);
  real q = qGlob;
  real rho_surf = rho_surfGlob;
  real rho_min = densityFloorGlob;
  real xcm = q / (1.0 + q);
  real potL1 = pot_L1Glob;
  real L1 = L1Glob;
  real Gamma = Gamma_Glob;
  real R_lobe = fabs(r1x - L1);
  real Rcut = ffactor* R_lobe;
  
 
  const real PRS_FLOOR = AdiabaticPressure(rho_min, rho_surf, cs2_ref, Gamma);

  for(int k = 0; k < d.np_tot[KDIR]; ++k) {
    for(int j = 0; j < d.np_tot[JDIR]; ++j) {
      for(int i = 0; i < d.np_tot[IDIR]; ++i) {
        real r = d.x[IDIR](i);
        real th = d.x[JDIR](j);
        real ph = d.x[KDIR](k);
        real X = r * sin(th) * cos(ph);
        real Y = r * sin(th) * sin(ph);
        real Z = r * cos(th);
        real pot = RochePotentialAtXYZ(X, Y, Z, q, xcm);
        real dx = X - r1x;
        real dist_M1 = sqrt(dx*dx + Y*Y + Z*Z);

        real rho_val;
        if (pot < potL1 && dist_M1 < fabs(Rcut)) {
          rho_val = fmax(rho_surf, rho_min);
        } else {
          rho_val = rho_min;
        }

        d.Vc(RHO,k,j,i) = rho_val;
        d.Vc(VX1,k,j,i) = 0.0;
        d.Vc(VX2,k,j,i) = 0.0;
        d.Vc(VX3,k,j,i) = 0.0;
        
     
        d.Vc(PRS,k,j,i) = AdiabaticPressure(rho_val, rho_surf, cs2_ref, Gamma);
      }
    }
  }
  d.SyncToDevice();
}

void MakeAnalysis(DataBlock & data) {}

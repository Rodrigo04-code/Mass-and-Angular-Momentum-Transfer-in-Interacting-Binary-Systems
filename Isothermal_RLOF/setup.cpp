/*
Author: Rodrigo Pérez San Martín
Project: Mass transfer in binary system
Description: This code implements the physical setup required to simulate mass transfer in a binary system using the Idefix hydrodynamic framework. The program defines a user-specified Roche gravitational potential in Cartesian coordinates and computes its representation on a spherical grid. It also enrolls the corresponding body force terms associated with rotation in the corotating frame. A custom internal boundary condition is applied to mimic an accretion sink near the primary object and to maintain an isothermal atmosphere around the donor star close to the L1 point. Initial conditions for density and velocity are generated based on the Roche potential to create a stable configuration prior to the onset of the overflow. The overall objective of the code is to provide a controlled numerical environment for studying disk formation and outflow structures resulting from Roche lobe overflow in close binary systems.
*/


#include "idefix.hpp"
#include "setup.hpp"
#include <cmath>


// Parameters


real qGlob;
real L1Glob;
real rho_surfGlob;
real densityFloorGlob;
real pot_L1Glob;
real cs2_glob;

const real soft_eps  = 0.01;
const real soft_eps2 = soft_eps * soft_eps;

const real r1x = 1.0;
const real r2x = 0.0;
const real ffactor = 1.1;

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

  idefix_for("Potential",
    0, data.np_tot[KDIR],
    0, data.np_tot[JDIR],
    0, data.np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {

      real r  = x1(i);
      real th = x2(j);
      real ph = x3(k);

      real X = r * sin(th) * cos(ph);
      real Y = r * sin(th) * sin(ph);
      real Z = r * cos(th);

      phi(k,j,i) = RochePotentialAtXYZ(X,Y,Z,q,xcm);
    });
}

// Coriolis Force
void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {

  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;
  real q = qGlob;

  idefix_for("BodyForce",
    data.beg[KDIR], data.end[KDIR],
    data.beg[JDIR], data.end[JDIR],
    data.beg[IDIR], data.end[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {

      real th = x2(j);
      real v1 = Vc(VX1,k,j,i);
      real v2 = Vc(VX2,k,j,i);
      real v3 = Vc(VX3,k,j,i);

      force(IDIR,k,j,i) =  2.0 * sqrt(1+q) * (v3 * sin(th));
      force(JDIR,k,j,i) =  2.0 * sqrt(1+q) * (v3 * cos(th));
      force(KDIR,k,j,i) = -2.0 * sqrt(1+q) * (v2 * cos(th) + v1 * sin(th));
    });
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

  real q     = qGlob;
  real xcm   = q / (1.0 + q);
  real L1    = L1Glob;
  real rho_s = rho_surfGlob;
  real rho_f = densityFloorGlob;
  real phiL1 = pot_L1Glob;

  real R_lobe = fabs(r1x - L1);
  real Rcut   = ffactor * R_lobe;
  real rmin   = 0.06;

  idefix_for("UserdefBoundary",
    0, data->np_tot[KDIR],
    0, data->np_tot[JDIR],
    0, data->np_tot[IDIR],
    KOKKOS_LAMBDA(int k, int j, int i) {

      real r  = x1(i);
      real th = x2(j);
      real ph = x3(k);

      real X = r * sin(th) * cos(ph);
      real Y = r * sin(th) * sin(ph);
      real Z = r * cos(th);

      real pot = RochePotentialAtXYZ(X,Y,Z,q,xcm);
      real dx  = X - r1x;
      real dist_M1 = sqrt(dx*dx + Y*Y + Z*Z);

      real rho = Vc(RHO,k,j,i);
      real v1  = Vc(VX1,k,j,i);
      real v2  = Vc(VX2,k,j,i);
      real v3  = Vc(VX3,k,j,i);

      // L1
      if (dist_M1 < Rcut && pot < phiL1) {
        rho = fmax(rho, rho_s);
        v1 = v2 = v3 = 0.0;
      }

      // rhoo floor
      if (rho < rho_f) rho = rho_f;

      // Sink interno
      if (r < rmin) {
        const real alpha = 0.2;
        rho = (1.0 - alpha) * rho + alpha * rho_f;
        v1  = (1.0 - alpha) * v1;
        v2  = (1.0 - alpha) * v2;
        v3  = (1.0 - alpha) * v3;
      }

      Vc(RHO,k,j,i) = rho;
      Vc(VX1,k,j,i) = v1;
      Vc(VX2,k,j,i) = v2;
      Vc(VX3,k,j,i) = v3;
    });
}

// Setup

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {

  data.gravity->EnrollPotential(&Potential);
  data.gravity->EnrollBodyForce(BodyForce);
  data.hydro->EnrollUserDefBoundary(&UserdefBoundary);

  rho_surfGlob     = input.Get<real>("Setup","rho_surf",0);
  qGlob            = input.Get<real>("Setup","q",0);
  L1Glob           = input.Get<real>("Setup","L1",0);
  densityFloorGlob = input.Get<real>("Setup","densityFloor",0);
  cs2_glob         = input.Get<real>("Hydro","csiso",1);

  real xcm = qGlob / (1.0 + qGlob);
  pot_L1Glob = RochePotentialAtXYZ(L1Glob, 0.0, 0.0, qGlob, xcm);
}

// Initial Conditions

void Setup::InitFlow(DataBlock &data) {

  DataBlockHost d(data);

  real q   = qGlob;
  real xcm = q / (1.0 + qGlob);

  real rho_s = rho_surfGlob;
  real rho_f = densityFloorGlob;
  real L1    = L1Glob;
  real phiL1 = pot_L1Glob;

  real R_lobe = fabs(r1x - L1);
  real Rcut   = ffactor * R_lobe;

  for(int k=0; k<d.np_tot[KDIR]; k++) {
    for(int j=0; j<d.np_tot[JDIR]; j++) {
      for(int i=0; i<d.np_tot[IDIR]; i++) {

        real r  = d.x[IDIR](i);
        real th = d.x[JDIR](j);
        real ph = d.x[KDIR](k);

        real X = r * sin(th) * cos(ph);
        real Y = r * sin(th) * sin(ph);
        real Z = r * cos(th);

        real pot = RochePotentialAtXYZ(X,Y,Z,q,xcm);
        real dx  = X - r1x;
        real dist_M1 = sqrt(dx*dx + Y*Y + Z*Z);

        real rho = (pot < phiL1 && dist_M1 < Rcut) ? rho_s : rho_f;

        d.Vc(RHO,k,j,i) = rho;
        d.Vc(VX1,k,j,i) = 0.0;
        d.Vc(VX2,k,j,i) = 0.0;
        d.Vc(VX3,k,j,i) = 0.0;
      }
    }
  }

  d.SyncToDevice();
}

void MakeAnalysis(DataBlock & data) {}

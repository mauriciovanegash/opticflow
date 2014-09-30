/*****************************************************************************/
/* --- fed ----------------------------------------------------------------- */
/* Compute time steps for Fast Explicit Diffusion (FED)                      */
/*                                                                           */
/* (C) 2013 Sven Grewenig*, Joachim Weickert*, Andres Bruhn^,                */
/*          and Pascal Gwosdek*                                              */
/*                                                                           */
/*          * Mathematical Image Analysis Group,                             */
/*            Saarland University, Germany                                   */
/*          ^ Intelligent Systems Group,                                     */
/*            Institute for Visualization and Interactive Systems            */
/*            University of Stuttgart, Germany                               */          
/*                                                                           */
/* This program is free software: you can redistribute it and/or modify it   */
/* under the terms of the GNU General Public License as published by the     */
/* Free Software Foundation, either version 3 of the License, or (at your    */
/* option) any later version.                                                */
/*                                                                           */
/* This program is distributed in the hope that it will be useful, but       */
/* WITHOUT ANY WARRANTY; without even the implied warranty of                */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General  */
/* Public License for more details.                                          */
/*                                                                           */
/* You should have received a copy of the GNU General Public License along   */
/* with this program. If not, see <http://www.gnu.org/licenses/>.            */
/*                                                                           */
/* If you intend to use this library for your own publications, please cite  */
/* Grewenig et al. (2010) as a reference for FED.                            */
/*                                                                           */
/* Version 1.06-L (2013-02-27)                                               */
/*****************************************************************************/

#ifndef FED_INCLUDED
#define FED_INCLUDED

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "fed_kappa.h"

/* ------------------------------------------------------------------------- */

/*****************************************************************************/
/* Allocates an array of n time steps and fills it with FED time step sizes, */
/* such that the maximal stopping time for this cycle is obtained.           */
/*                                                                           */
/* RETURNS n if everything is ok, or 0 on failure.                           */
/*****************************************************************************/
int fed_tau_by_steps
(
  int           n,              /* > Desired number of internal steps        */
  float         tau_max,        /* > Stability limit for explicit (0.5^Dim)  */
  int           reordering,     /* > Reordering flag                         */
  float         **tau           /* < Time step widths (allocated inside)     */
);

/* ------------------------------------------------------------------------- */

/*****************************************************************************/
/* Allocates an array of the least number of time steps such that a certain  */
/* stopping time per cycle can be obtained, and fills it with the respective */
/* FED time step sizes.                                                      */
/*                                                                           */
/* RETURNS number of time steps per cycle, or 0 on failure.                  */
/*****************************************************************************/
int fed_tau_by_cycle_time
(
  float         t,              /* > Desired cycle stopping time             */
  float         tau_max,        /* > Stability limit for explicit (0.5^Dim)  */
  int           reordering,     /* > Reordering flag                         */
  float         **tau           /* < Time step widths (allocated inside)     */
);

/* ------------------------------------------------------------------------- */

/*****************************************************************************/
/* Allocates an array of the least number of time steps such that a certain  */
/* stopping time for the whole process can be obtained, and fills it with    */
/* the respective FED time step sizes for one cycle.                         */
/*                                                                           */
/* RETURNS number of time steps per cycle, or 0 on failure.                  */
/*****************************************************************************/
int fed_tau_by_process_time
(
  float         T,              /* > Desired process stopping time           */
  int           M,              /* > Desired number of cycles                */
  float         tau_max,        /* > Stability limit for explicit (0.5^Dim)  */
  int           reordering,     /* > Reordering flag                         */
  float         **tau           /* < Time step widths (allocated inside)     */
);

/* ------------------------------------------------------------------------- */

/*****************************************************************************/
/* Computes the maximal cycle time that can be obtained using a certain      */
/* number of steps. This corresponds to the cycle time that arises from a    */
/* tau array which has been created using fed_tau_by_steps.                  */
/*                                                                           */
/* RETURNS cycle time t                                                      */
/*****************************************************************************/
float fed_max_cycle_time_by_steps
(
  int           n,              /* > Number of steps per FED cycle           */
  float         tau_max         /* > Stability limit for explicit (0.5^Dim)  */
);

/* ------------------------------------------------------------------------- */

/*****************************************************************************/
/* Computes the maximal process time that can be obtained using a certain    */
/* number of steps. This corresponds to the cycle time that arises from a    */
/* tau array which has been created using fed_tau_by_steps.                  */
/*                                                                           */
/* RETURNS cycle time t                                                      */
/*****************************************************************************/
float fed_max_process_time_by_steps
(
  int           n,              /* > Number of steps per FED cycle           */
  int           M,              /* > Number of cycles                        */
  float         tau_max         /* > Stability limit for explicit (0.5^Dim)  */
);

/* ------------------------------------------------------------------------- */

/*****************************************************************************/
/* Allocates an array of n relaxation parameters and fills it with the FED   */
/* based parameters for Fast-Jacobi.                                         */
/*                                                                           */
/* RETURNS n if everything is ok, or 0 on failure.                           */
/*****************************************************************************/
int fastjac_relax_params
(
  int           n,              /* > Cycle length                            */
  float         omega_max,      /* > Stability limit for Jacobi over-relax.  */
  int           reordering,     /* > Reordering flag                         */
  float         **omega         /* < Relaxation parameters (allocated inside)*/
);

#endif

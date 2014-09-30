/*****************************************************************************/
/* --- fed ----------------------------------------------------------------- */
/* Compute time steps for Fast Explicit Diffusion (FED)                      */
/* and relaxation parameters for Fast-Jacobi (FJ)                            */
/*                                                                           */
/* (C) 2013 Sven Grewenig*, Joachim Weickert*, Andres Bruhn^,                */
/*          Pascal Gwosdek*, and Christopher Schroers*                       */
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

#include "fed.h"

/*****************************************************************************/
/* Forward declarations of private functions (INTERNAL)                      */
/*****************************************************************************/
int _fed_tau_internal(int, float, float, int, float**);
int _fed_is_prime_internal(int);

/* ------------------------------------------------------------------------- */

/*****************************************************************************/
/* INTERNAL ---------------------------------------------------------------- */
/*                                                                           */
/* Uses Eratosthenes' Sieve to check if a number is prime.                   */
/* RETURNS if number is a prime                                              */
/*****************************************************************************/
int _fed_is_prime_internal
(
  int           number          /* > Number to check                         */
)
{
  int i;
  int s = (int)sqrtf((float)number + 1.0f);

  for (i = 2; i <= s; ++i)
    if (!(number % i))
      return 0;

  return 1;
}

/* ------------------------------------------------------------------------- */

/*****************************************************************************/
/* INTERNAL ---------------------------------------------------------------- */
/*                                                                           */
/* Allocates an array of time steps and fills it with FED time step sizes.   */
/* RETURNS n if everything is ok, or 0 on failure.                           */
/*****************************************************************************/
int _fed_tau_internal
(
  int           n,              /* > Number of internal steps                */
  float         scale,          /* > Ratio of t we search to maximal t       */
  float         tau_max,        /* > Stability limit for explicit scheme     */
  int           reordering,     /* > Reordering flag                         */
  float         **tau           /* < Time step widths (allocated inside)     */
)
{
  int   k;          /* Loop counter                                          */
  float c, d;       /* Time savers                                           */

  float *tauh = 0;  /* Helper array for unsorted taus                        */

  #ifndef NDEBUG
  if (n <= 0)
    return 0;
  #endif

  /* Allocate memory for the time step sizes                                 */
  *tau = (float*)malloc(n * sizeof(float));

  if (reordering)
    tauh = (float*)malloc(n * sizeof(float));

  if (!(*tau) || (reordering && !tauh))
  {
    #ifndef NDEBUG
    printf("ERROR: Allocation of FED time step array failed.\n");
    #endif

    if ((*tau))
      free(*tau);
    if (tauh)
      free(tauh);

    return 0;
  }

  /* Compute time saver                                                      */
  c = 1.0f / (4.0f * (float)n + 2.0f);
  d = scale * tau_max / 2.0f;

  /* Set up originally ordered tau vector                                    */
  for (k = 0; k < n; ++k)
  {
    float h = cosf(M_PI * (2.0f * (float)k + 1.0f) * c);

    if (reordering)
      tauh[k] = d / (h * h);
    else
      (*tau)[k] = d / (h * h);
  }

  /* Permute list of time steps according to chosen reordering function      */
  int kappa, prime, l;

  if (reordering)
  {
    if (n > FED_MAXKAPPA)
    {
      #ifndef NDEBUG
      printf("ERROR: Optimal kappa not in lookup table. "
             "Trying generic approach.\n");
      #endif
      kappa = n / 4;
    }
    else
      kappa = fed_kappalookup[n];

    /* Get modulus for permutation                                         */
    prime = n + 1;
    while (!_fed_is_prime_internal(prime))
      prime++;

    /* Perform permutation                                                 */
    for (k = 0, l = 0; l < n; ++k, ++l)
    {
      int index;
      while ((index = ((k+1)*kappa) % prime - 1) >= n)
        k++;

      (*tau)[l] = tauh[index];
    }
  }

  if (reordering)
    free(tauh);

  return n;
}

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
  float         tau_max,        /* > Stability limit for explicit scheme     */
  int           reordering,     /* > Reordering flag                         */
  float         **tau           /* < Time step widths (allocated inside)     */
)
{
  /* Call internal FED time step creation routine with maximal stopping time */
  return _fed_tau_internal(n, 1.0f, tau_max, reordering, tau);
}

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
  float         tau_max,        /* > Stability limit for explicit scheme     */
  int           reordering,     /* > Reordering flag                         */
  float         **tau           /* < Time step widths (allocated inside)     */
)
{
  int   n;      /* Number of time steps                                      */
  float scale;  /* Ratio of t we search to maximal t                         */

  /* Compute necessary number of time steps                                  */
  n     = (int)(ceilf(sqrtf(3.0 * t / tau_max + 0.25f) - 0.5f - 1.0e-8f)
                + 0.5f);
  scale = 3.0 * t / (tau_max * (float)(n * (n + 1)));

  /* Call internal FED time step creation routine                            */
  return _fed_tau_internal(n, scale, tau_max, reordering, tau);
}

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
  float         tau_max,        /* > Stability limit for explicit scheme     */
  int           reordering,     /* > Reordering flag                         */
  float         **tau           /* < Time step widths (allocated inside)     */
)
{
  /* All cycles have the same fraction of the stopping time                  */
  return fed_tau_by_cycle_time(T/(float)M, tau_max, reordering, tau);
}

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
  float         tau_max         /* > Stability limit for explicit scheme     */
)
{
  return (tau_max * (float)(n * (n + 1))) / 3.0f;
}

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
  float         tau_max         /* > Stability limit for explicit scheme     */
)
{
  return (tau_max * (float)(n * (n + 1) * M)) / 3.0f;
}

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
)
{
  /* Call internal FED time step creation routine */
  return _fed_tau_internal(n, 1.0f, omega_max, reordering, omega);
}

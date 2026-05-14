#include <cmath>
#include <vector>
#include <complex> // for the jacobian
#include <boost/preprocessor/seq/for_each.hpp>

#include "ADTypes.hpp"

#include "physics.h"
#include "real_gas.h"
#include "navier_stokes_real_gas.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nspecies, int nstate, typename real>
NavierStokes_RealGas <dim, nspecies, nstate, real>::NavierStokes_RealGas ( 
    const Parameters::AllParameters *const                    parameters_input,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type)
    : RealGas<dim,nspecies,nstate,real>(parameters_input) 
    , viscosity_coefficient_inf(1.0) // Nondimensional - Free stream values
    , use_constant_viscosity(use_constant_viscosity)
    , constant_viscosity(constant_viscosity) // Nondimensional - Free stream values
    , prandtl_number(prandtl_number)
    , reynolds_number_inf(reynolds_number_inf)
    , isothermal_wall_temperature(isothermal_wall_temperature) // Nondimensional - Free stream values
    , thermal_boundary_condition_type(thermal_boundary_condition_type)
    , sutherlands_temperature(110.4) // Sutherland's temperature. Units: [K]
    , freestream_temperature(temperature_inf) // Freestream temperature. Units: [K]
    , temperature_ratio(sutherlands_temperature/freestream_temperature)
{
    static_assert(nstate==dim+nspecies+1, "Physics::NavierStokes_RealGas() should be created with nstate=dim+nspecies+1");
    // Nothing to do here so far
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokes_RealGas<dim,nspecies,nstate,real>
::convert_conservative_gradient_to_primitive_gradient (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // conservative_soln_gradient is solution_gradient
    std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient;

    // get primitive solution
    const std::array<real,nstate> primitive_soln = this->template convert_conservative_to_primitive(conservative_soln); // from Euler

    // extract from primitive solution
    const real density = primitive_soln[0];
    const dealii::Tensor<1,dim,real> vel = this->template extract_velocities_from_primitive(primitive_soln); // from Euler

    // mass fractions
    const std::array<real,nspecies> mass_fractions = this->template compute_mass_fractions(conservative_soln);

     // mixture density gradient
    for (int d=0; d<dim; d++) {
        primitive_soln_gradient[0][d] = conservative_soln_gradient[0][d];
    }

    // velocities gradient
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            primitive_soln_gradient[1+d1][d2] = (conservative_soln_gradient[1+d1][d2] - vel[d1]*conservative_soln_gradient[0][d2])/density;
        }        
    }

    // mass fraction gradient 
    for (int d1=dim+2; d1<dim+2+(nspecies-1); d1++) {
        for (int d2=0; d2<dim; d2++) {
            primitive_soln_gradient[d1][d2] = (conservative_soln_gradient[d1][d2] - primitive_soln[d1]*conservative_soln_gradient[0][d2])/density;
        }
    }

    const dealii::Tensor<1,dim,real> temperature_gradient = compute_temperature_gradient(conservative_soln, conservative_soln_gradient);
    
    const real temperature = this->template compute_temperature(conservative_soln);

    real mixture_gas_constant = 0.0;
    for (int s=0; s<nspecies; s++) {
        mixture_gas_constant += mass_fractions[s] * this->Rs[s];
    }

    const std::array<dealii::Tensor<1,dim,real>, nspecies> grad_Y = compute_mass_fraction_gradients_from_primitive_gradient(primitive_soln_gradient);

    for (int d=0; d<dim; d++) {
        real grad_Rmix = 0.0;
        for (int s=0; s<nspecies; s++) {
            grad_Rmix += this->Rs[s] * grad_Y[s][d];
        }

        primitive_soln_gradient[dim+1][d] = (mixture_gas_constant * temperature * primitive_soln_gradient[0][d] + density * temperature * grad_Rmix + density * mixture_gas_constant * temperature_gradient[d]) / (this->gam_ref * this->mach_ref_sqr);
    }

    return primitive_soln_gradient;

}

// TEMPERATURE GRADIENT
template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<1,dim,real> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_temperature_gradient(const std::array<real, nstate> &conservative_soln, const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    const real density = conservative_soln[0];
    const dealii::Tensor<1,dim,real> vel = this->template compute_velocities(conservative_soln);
    const real temperature = this->template compute_temperature(conservative_soln);
    const std::array<real,nspecies> e_s = this->template compute_species_specific_internal_energy(temperature);
    const std::array<real, nspecies> Cv_s = this->template compute_species_specific_Cv(temperature);
    const std::array<real, nspecies> mass_fractions = this->template compute_mass_fractions(conservative_soln);

    real Cv_mix = 0.0;
    for (int s=0; s<nspecies; s++) {
        Cv_mix += mass_fractions[s] * Cv_s[s];
    }

    const real E = conservative_soln[dim+1] / density;

    // velocities gradient
    std::array<dealii::Tensor<1,dim,real>,dim> vel_gradient;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            vel_gradient[d1][d2] = (conservative_soln_gradient[1+d1][d2] - vel[d1]*conservative_soln_gradient[0][d2])/density;
        }        
    }

    // mass fraction gradient 
    std::array<dealii::Tensor<1,dim,real>,nspecies> grad_Y;
    for (int s=0; s<nspecies-1; s++) {
        for (int d=0; d<dim; d++) {
            grad_Y[s][d] = (conservative_soln_gradient[dim+2+s][d] - mass_fractions[s]*conservative_soln_gradient[0][d])/density;
        }
    }

    //last speices
    for (int d=0; d<dim; d++) {
        grad_Y[nspecies-1][d] = 0.0;
        for (int s=0; s<nspecies-1; s++) {
            grad_Y[nspecies-1][d] -= grad_Y[s][d];
        }
    }

    dealii::Tensor<1,dim,real> temperature_gradient;
    for (int d=0; d<dim; d++) {
        const real grad_E = (conservative_soln_gradient[dim+1][d] - E*conservative_soln_gradient[0][d]) / density;

        real vel_gradvel = 0.0;
        for (int d2=0; d2<dim; d2++) {
            vel_gradvel += vel[d2] * vel_gradient[d2][d];
        }
        
        real e_s_gradYs = 0.0;
        for (int s=0; s<nspecies; s++) {
            e_s_gradYs += e_s[s] * grad_Y[s][d];
        }

        temperature_gradient[d] = (grad_E - vel_gradvel - e_s_gradYs) / Cv_mix * this->gam_ref * this->mach_ref_sqr;

    }

    return temperature_gradient;
}

// template <int dim, int nspecies, int nstate, typename real>
// std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokes_RealGas<dim,nspecies,nstate,real>
// ::convert_conservative_gradient_to_primitive_gradient (
//     const std::array<real,nstate> &conservative_soln,
//     const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
// {
//     // conservative_soln_gradient is solution_gradient
//     std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient;

//     // get primitive solution
//     const std::array<real,nstate> primitive_soln = this->template convert_conservative_to_primitive(conservative_soln); // from real gas

//     // extract from primitive solution
//     const real density = primitive_soln[0];
//     const dealii::Tensor<1,dim,real> vel = this->template extract_velocities_from_primitive(primitive_soln); // from real gas

//      // mixture density gradient
//     for (int d=0; d<dim; d++) {
//         primitive_soln_gradient[0][d] = conservative_soln_gradient[0][d];
//     }

//     // velocities gradient
//     for (int d1=0; d1<dim; d1++) {
//         for (int d2=0; d2<dim; d2++) {
//             primitive_soln_gradient[1+d1][d2] = (conservative_soln_gradient[1+d1][d2] - vel[d1]*conservative_soln_gradient[0][d2])/density;
//         }        
//     }

//     // mass fraction gradient 
//     for (int d1=dim+2; d1<dim+2+(nspecies-1); d1++) {
//         for (int d2=0; d2<dim; d2++) {
//             primitive_soln_gradient[d1][d2] = (conservative_soln_gradient[d1][d2] - primitive_soln[d1]*conservative_soln_gradient[0][d2])/density;
//         }
//     }

// //     // pressure gradient
// //     // -- formulation 1:
// //     // const double vel2 = this->template compute_velocity_squared(vel); // from Euler
// //     // for (int d1=0; d1<dim; d1++) {
// //     //     primitive_soln_gradient[nstate-1][d1] = conservative_soln_gradient[nstate-1][d1] - 0.5*vel2*conservative_soln_gradient[0][d1];
// //     //     for (int d2=0; d2<dim; d2++) {
// //     //         primitive_soln_gradient[nstate-1][d1] -= conservative_soln[1+d2]*primitive_soln_gradient[1+d2][d1];
// //     //     }
// //     //     primitive_soln_gradient[nstate-1][d1] *= this->gamm1;
// //     // }
// //     // -- formulation 2 (equivalent to formulation 1):
//     for (int d1=0; d1<dim; d1++) {
//         primitive_soln_gradient[dim+1][d1] = conservative_soln_gradient[dim+1][d1];
//         for (int d2=0; d2<dim; d2++) {
//             primitive_soln_gradient[dim+1][d1] -= 0.5*(primitive_soln[1+d2]*conservative_soln_gradient[1+d2][d1]  
//                                                            + conservative_soln[1+d2]*primitive_soln_gradient[1+d2][d1]);
//         }
//         primitive_soln_gradient[dim+1][d1] *= this->gamm1;
//     }
//     return primitive_soln_gradient;
// }

// template <int dim, int nspecies, int nstate, typename real>
// dealii::Tensor<1,dim,real> NavierStokes_RealGas<dim,nspecies,nstate,real>
// ::compute_temperature_gradient (
//     const std::array<real,nstate> &primitive_soln,
//     const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
// {
//     const real density = primitive_soln[0];
//     const real temperature = this->template compute_temperature(this->template convert_primitive_to_conservative(primitive_soln)); // from Real Gas

//     dealii::Tensor<1,dim,real> temperature_gradient;
//     for (int d=0; d<dim; d++) {
//         temperature_gradient[d] = (this->gam_ref*this->mach_ref_sqr*primitive_soln_gradient[dim+1][d] - temperature*primitive_soln_gradient[0][d])/density;
//     }
//     return temperature_gradient;
// }

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_viscosity_coefficient (const std::array<real,nstate> &primitive_soln) const
{   
    // Use either Wile's mixture law or constant viscosity
    real viscosity_coefficient;
    if(use_constant_viscosity){
        viscosity_coefficient = 1.0*constant_viscosity;
    } else {
        viscosity_coefficient = compute_mixture_viscosity_coefficient_wilkes_rule(primitive_soln);
    }

    return viscosity_coefficient;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_viscosity_coefficient_sutherlands_law (const std::array<real,nstate> &primitive_soln, const int species_index) const
{
    /* Nondimensionalized viscosity coefficient, \mu^{*}
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     * Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    const real temperature = this->template compute_temperature(this->template convert_primitive_to_conservative(primitive_soln)); // from Real Gas, dimensionless

    const real sutherlands_temperature_k = this->species_sutherland_temperature[species_index];
    const real T_inf = this->temperature_ref;
    
    const real temperature_ratio = sutherlands_temperature_k / T_inf;
    //const real viscosity_ratio = this->species_mu_ref[species_index] / viscosity_coefficient_inf ;

    const real viscosity_coefficient = ((1.0 + temperature_ratio)/(temperature + temperature_ratio))*pow(temperature,1.5);
    
    return viscosity_coefficient;
}
  
template <int dim, int nspecies, int nstate, typename real>
std::array<real, nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_mole_fractions(const std::array<real, nstate> &primitive_soln) const
{
    /* Added by Clara
     * Computes the mole fraction of each species and stores it in an array.
    */

    std::array<real, nspecies> mass_fractions;
    real sum_mass_fractions = 0.0;
    for (int s = 0; s < nspecies-1; s++) {
        mass_fractions[s] = primitive_soln[dim + 2 + s];
        sum_mass_fractions += mass_fractions[s];
    }
    mass_fractions[nspecies-1] = 1.0 - sum_mass_fractions;

    real den = 0.0;
    for (int s = 0; s < nspecies; s++) {
        den += mass_fractions[s] / this->species_weight[s];
    }

    std::array<real, nspecies> mole_fractions;
    for (int s = 0; s < nspecies; s++) {
        mole_fractions[s] = (mass_fractions[s] / this->species_weight[s]) / den;
    }

    return mole_fractions;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<real, nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_mass_fractions_from_primitive(const std::array<real, nstate> &primitive_soln) const
{
    std::array<real, nspecies> mass_fractions;
    real sum_mass_fractions = 0.0;

    for (int s=0; s<nspecies-1; s++) {
        mass_fractions[s] = primitive_soln[dim+2+s];
        sum_mass_fractions += mass_fractions[s];
    }
    mass_fractions[nspecies-1] = 1.0 - sum_mass_fractions;

    return mass_fractions;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>, nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_mass_fraction_gradients_from_primitive_gradient(const std::array<dealii::Tensor<1,dim,real>, nstate> &primitive_soln_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>, nspecies> mass_fraction_gradients;

    for (int s=0; s<nspecies-1; s++) {
        mass_fraction_gradients[s] = primitive_soln_gradient[dim+2+s];
    }

    mass_fraction_gradients[nspecies-1] = 0.0;
    for (int s=0; s<nspecies-1; s++) {
        mass_fraction_gradients[nspecies-1] -= mass_fraction_gradients[s];
    }

    return mass_fraction_gradients;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1, dim, real>, nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_mole_fraction_gradients(
        const std::array<real, nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>, nstate> &primitive_soln_gradient) const
{
    const std::array<real, nspecies> mass_fractions = compute_mass_fractions_from_primitive(primitive_soln);
    const std::array<dealii::Tensor<1,dim,real>, nspecies> mass_fraction_gradients = compute_mass_fraction_gradients_from_primitive_gradient(primitive_soln_gradient);

    // sum = sum of the mass fractions over the species weight for all species 
    real sum = 0.0;
    for (int s=0; s<nspecies; s++) {
        sum += mass_fractions[s] / this->species_weight[s];
    }

    dealii::Tensor<1,dim,real> gradient_sum;
    gradient_sum = 0.0;
    for (int s=0; s<nspecies; s++) {
        for (int d=0; d<dim; d++) {
            gradient_sum[d] += mass_fraction_gradients[s][d] / this->species_weight[s];
        } 
    }

    std::array<dealii::Tensor<1,dim,real>, nspecies> mole_fraction_gradients;
    for (int s=0; s<nspecies; s++) {
        const real Y_over_W = mass_fractions[s] / this->species_weight[s];
        const real factor = Y_over_W / (sum*sum);
        const real inverse_species_weight = 1.0 / this->species_weight[s];

        for (int d=0; d<dim; d++) {
            mole_fraction_gradients[s][d] = (mass_fraction_gradients[s][d] * inverse_species_weight) / sum - factor * gradient_sum[d];
        }
    }

    return mole_fraction_gradients;

}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1, dim, real>, nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_diffusion_driving_forces(
        const std::array<real, nstate> &primitive_soln, 
        const std::array<dealii::Tensor<1,dim,real>, nstate> &primitive_soln_gradient) const
{
    const std::array<real, nspecies> mass_fractions = compute_mass_fractions_from_primitive(primitive_soln);
    const std::array<real, nspecies> mole_fractions = compute_mole_fractions(primitive_soln);
    const std::array<dealii::Tensor<1,dim,real>, nspecies> mole_fraction_gradients = compute_mole_fraction_gradients(primitive_soln, primitive_soln_gradient);
    const real pressure = primitive_soln[dim+1];
    const dealii::Tensor<1,dim,real> pressure_gradient = primitive_soln_gradient[dim+1];

    dealii::Tensor<1,dim,real> gradient_ln_P;
    for (int d=0; d<dim; d++) {
        gradient_ln_P[d] = pressure_gradient[d] / pressure;
    }

    std::array<dealii::Tensor<1,dim,real>,nspecies> diffusion_driving_force;
    for (int s=0; s<nspecies; s++) {
        const real mole_fractions_minus_mass_fractions = mole_fractions[s] - mass_fractions[s];

        for (int d=0; d<dim; d++) {
            diffusion_driving_force[s][d] = mole_fraction_gradients[s][d] + mole_fractions_minus_mass_fractions * gradient_ln_P[d];
        }
    }

    return diffusion_driving_force;
    
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>, nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_nondimensional_diffusion_driving_forces(
        const std::array<real, nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>, nstate> &primitive_soln_gradient) const
{
    // Nondimensionalize using d_k^* = L_ref * d_k.
    std::array<dealii::Tensor<1,dim,real>, nspecies> non_dimensional_diffusion_driving_forces = compute_diffusion_driving_forces(primitive_soln, primitive_soln_gradient);

    for (int s = 0; s < nspecies; s++) {
        non_dimensional_diffusion_driving_forces[s] *= this->ref_length;
    }

    return non_dimensional_diffusion_driving_forces;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_collision_integral(const real reduced_temperature) const
{
    const real collision_integral = 1.0548 * pow(reduced_temperature, -0.15504) + pow(reduced_temperature + 0.55909, -2.1705);

    return collision_integral;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_reduced_molecular_weight(const int j, const int k) const 
{
    const real molecular_weight_j = this->species_weight[j];
    const real molecular_weight_k = this->species_weight[k];

    return (molecular_weight_j * molecular_weight_k) / (molecular_weight_j + molecular_weight_k);
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_reduced_collision_diameter(const int j, const int k) const
{
    const real collision_diameter_j = this->species_collision_diameter[j]*pow(10,0);
    const real collision_diameter_k = this->species_collision_diameter[k]*pow(10,0);

    return 0.5 * (collision_diameter_j + collision_diameter_k);
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_reduced_temperature(const real T_K, const int j, const int k) const 
{
    const real boiling_temperature_j = this->species_boiling_temperature[j];
    const real boiling_temperature_k = this->species_boiling_temperature[k];

    return T_K / (1.15 * sqrt(boiling_temperature_j * boiling_temperature_k));
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_dimensional_pressure(const std::array<real, nstate> &primitive_soln) const 
{
    const real non_dimensional_pressure = primitive_soln[dim+1];
    const real p_ref = (this->gam_ref * this->mach_ref_sqr) * (this->density_ref * this->R_ref * this->temperature_ref);

    return non_dimensional_pressure * p_ref;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_binary_diffusion_coefficient(
        const real T_K,
        const real P_Pa, 
        const int j, 
        const int k) const 
{
    if (j==k) return 0.0;

    const real W_jk = compute_reduced_molecular_weight(j, k);
    const real sigma_jk = compute_reduced_collision_diameter(j, k);
    const real Tstar_jk = compute_reduced_temperature(T_K, j, k);
    const real omegastar = compute_collision_integral(Tstar_jk);

    const real num = 5.95e-4 * sqrt(pow(T_K, 3) / W_jk);
    const real den = P_Pa * (sigma_jk * sigma_jk) * omegastar;
    const real binary_diffusion_coefficient = num / den;

    return binary_diffusion_coefficient;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<std::array<real, nspecies>, nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_binary_diffusion_matrix(const std::array<real, nstate> &primitive_soln) const 
{
    const real non_dimensional_temperature = this->template compute_temperature(this->template convert_primitive_to_conservative(primitive_soln));
    const real T_K = this->template compute_dimensional_temperature(non_dimensional_temperature);
    const real P_Pa = compute_dimensional_pressure(primitive_soln);

    std::array<std::array<real, nspecies>, nspecies> binary_diffusion_matrix;
    for (int s1=0; s1<nspecies; s1++) {
        for (int s2=0; s2<nspecies; s2++) {
            binary_diffusion_matrix[s1][s2] = 0.0;
        }
    }

    for (int s1=0; s1<nspecies; s1++) {
        for (int s2=s1+1; s2<nspecies; s2++) {
            const real binary_diffusion_coefficient = compute_binary_diffusion_coefficient(T_K, P_Pa, s1, s2);
            binary_diffusion_matrix[s1][s2] = binary_diffusion_coefficient;
            binary_diffusion_matrix[s2][s1] = binary_diffusion_coefficient;
        }
    }

    return binary_diffusion_matrix;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<std::array<real, nspecies>, nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_nondimensional_binary_diffusion_matrix(const std::array<real, nstate> &primitive_soln) const 
{
    std::array<std::array<real, nspecies>, nspecies> non_dimensional_binary_diffusion_matrix = compute_binary_diffusion_matrix(primitive_soln);

    const real D_ref = this->u_ref * this->ref_length;

    for (int s1=0; s1<nspecies; s1++) {
        for (int s2=0; s2<nspecies; s2++) {
            non_dimensional_binary_diffusion_matrix[s1][s2] /= D_ref;
        }
    }

    return non_dimensional_binary_diffusion_matrix;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_species_diffusion_flux(
        const std::array<real, nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>, nstate> &primitive_soln_gradient) const
{
    const real mixture_density = primitive_soln[0];

    const std::array<real, nspecies> mass_fractions = compute_mass_fractions_from_primitive(primitive_soln);
    const std::array<dealii::Tensor<1,dim,real>, nspecies> non_dimensional_diffusion_driving_forces = compute_nondimensional_diffusion_driving_forces(primitive_soln, primitive_soln_gradient);
    const std::array<std::array<real, nspecies>, nspecies> non_dimensional_binary_diffusion_matrix = compute_nondimensional_binary_diffusion_matrix(primitive_soln);

    std::array<dealii::Tensor<1,dim,real>, nspecies> species_diffusion_flux;
    for (int s=0; s<nspecies; s++) {
        for (int d=0; d<dim; d++) {
            species_diffusion_flux[s][d]=0.0;
        }
    }

    for (int s1=0; s1<nspecies-1; s1++) {
        dealii::Tensor<1,dim,real> sum;
        for (int d=0; d<dim; d++) {
            sum[d] = 0.0;
        }

        for (int s2=0; s2<nspecies; s2++) {
            for (int d=0; d<dim; d++) {
                sum[d] += non_dimensional_binary_diffusion_matrix[s1][s2] * non_dimensional_diffusion_driving_forces[s2][d];
            }
        }

        const real mixture_density_times_mass_fractions = -mixture_density * mass_fractions[s1];
        for (int d=0; d<dim; d++) {
            species_diffusion_flux[s1][d] =  mixture_density_times_mass_fractions * sum[d];
        }
    }

    for (int d=0; d<dim; d++) {
        species_diffusion_flux[nspecies-1][d] = 0.0;
    }
    for (int s1 = 0; s1<nspecies-1; s1++) {
        for (int d=0; d<dim; d++) {
            species_diffusion_flux[nspecies-1][d] -= species_diffusion_flux[s1][d];
        }
    }

    return species_diffusion_flux;

}

template <int dim, int nspecies, int nstate, typename real>
std::array<real, nspecies> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_species_viscosity_coefficients(const std::array<real, nstate> &primitive_soln) const
{
    std::array<real, nspecies> species_viscosity_coefficients;
    for (int s = 0; s < nspecies; s++) {
        species_viscosity_coefficients[s] = compute_viscosity_coefficient_sutherlands_law(primitive_soln,s);
    }
    return species_viscosity_coefficients;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_phi_kj(const int k, const int j, const std::array<real, nspecies> &species_viscosity_coefficients) const
{
    /* Phi term used in Wilke's rule.
     * Reference:
    */
    const real viscosity_coefficient_species_k = species_viscosity_coefficients[k];
    const real viscosity_coefficient_species_j = species_viscosity_coefficients[j];

    const real num = 1.0 + sqrt(viscosity_coefficient_species_k / viscosity_coefficient_species_j) * pow(this->species_weight[k] / this->species_weight[j], 0.25);
    const real num_sqr = num * num;

    const real den = sqrt(8) * sqrt(1 + this->species_weight[k] / this->species_weight[j]);

    const real phi_kj = num_sqr / den;

    return phi_kj;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_mixture_viscosity_coefficient_wilkes_rule(const std::array<real, nstate> &primitive_soln) const
{
    const std::array<real, nspecies> mole_fractions = compute_mole_fractions(primitive_soln);
    const std::array<real, nspecies> species_viscosity_coefficients = compute_species_viscosity_coefficients(primitive_soln);

    real mixture_viscosity_coefficient = 0.0;

    for (int s1 = 0; s1 < nspecies; s1++) {
        real sum = 0.0;
        for (int s2 = 0; s2 < nspecies; s2++) {
            if (s1 == s2) continue;
            sum += mole_fractions[s2] * compute_phi_kj(s1, s2, species_viscosity_coefficients);
        }

        const real den = 1.0 + (1 / mole_fractions[s1]) * sum;
               
        mixture_viscosity_coefficient += species_viscosity_coefficients[s1] / den;
    }
    return mixture_viscosity_coefficient;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim,nspecies,nstate,real>
::scale_viscosity_coefficient (const real viscosity_coefficient) const
{
    /* Scaled nondimensionalized viscosity coefficient, $\hat{\mu}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    const real scaled_viscosity_coefficient = viscosity_coefficient/reynolds_number_inf;
    
    return scaled_viscosity_coefficient;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_scaled_viscosity_coefficient (const std::array<real,nstate> &primitive_soln) const
{
    /* Scaled nondimensionalized viscosity coefficient, $\hat{\mu}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    const real viscosity_coefficient = compute_viscosity_coefficient(primitive_soln);
    const real scaled_viscosity_coefficient = scale_viscosity_coefficient(viscosity_coefficient);

    return scaled_viscosity_coefficient;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number (const real scaled_viscosity_coefficient, const real prandtl_number_input, const std::array<real,nstate> &conservative_soln) const
{
    /* Scaled nondimensionalized heat conductivity, $\hat{\kappa}^{*}$, given the scaled viscosity coefficient
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */

    const real temperature = this->compute_temperature(conservative_soln);
    const std::array<real,nspecies> mass_fractions = this->compute_mass_fractions(conservative_soln);
    const std::array<real,nspecies> Cp = this->compute_species_specific_Cp(temperature);
    const real mixture_Cp = this->compute_mixture_from_species(mass_fractions,Cp);

    const real scaled_heat_conductivity = scaled_viscosity_coefficient * mixture_Cp /(this->gam_ref*this->mach_ref_sqr*prandtl_number_input);
    
    return scaled_heat_conductivity;
}

template <int dim, int nspecies, int nstate, typename real>
inline real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_scaled_heat_conductivity (const std::array<real,nstate> &primitive_soln) const
{
    /* Scaled nondimensionalized heat conductivity, $\hat{\kappa}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */

    const std::array<real,nstate> conservative_soln = this->convert_primitive_to_conservative(primitive_soln);
    const real scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient(primitive_soln);

    const real scaled_heat_conductivity = compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number(scaled_viscosity_coefficient,prandtl_number, conservative_soln);
    
    return scaled_heat_conductivity;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<1,dim,real> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_heat_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    /* Nondimensionalized heat flux, $\bm{q}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    
    const std::array<real, nstate> primitive_soln = this->template convert_conservative_to_primitive(conservative_soln);
    const real scaled_heat_conductivity = compute_scaled_heat_conductivity(primitive_soln);
    const dealii::Tensor<1,dim,real> temperature_gradient = compute_temperature_gradient(conservative_soln, conservative_soln_gradient);
    // Compute the heat flux
    const dealii::Tensor<1,dim,real> heat_flux = compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient(scaled_heat_conductivity,temperature_gradient);
    return heat_flux;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<1,dim,real> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient (
    const real scaled_heat_conductivity,
    const dealii::Tensor<1,dim,real> &temperature_gradient) const
{
    /* Nondimensionalized heat flux, $\bm{q}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    dealii::Tensor<1,dim,real> heat_flux;
    for (int d=0; d<dim; d++) {
        heat_flux[d] = -scaled_heat_conductivity * temperature_gradient[d];
    }
    return heat_flux;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<1, dim, real> NavierStokes_RealGas<dim, nspecies, nstate, real>
    ::compute_total_heat_flux(
        const std::array<real, nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>, nstate> &conservative_soln_gradient,
        const std::array<dealii::Tensor<1,dim,real>, nspecies> &species_diffusion_flux) const
{
    dealii::Tensor<1,dim,real> total_heat_flux = compute_heat_flux(conservative_soln, conservative_soln_gradient);

    const real temperature = this->template compute_temperature(conservative_soln);
    const std::array<real, nspecies> species_enthalpy = this->template compute_species_specific_enthalpy(temperature);

    for (int s=0; s<nspecies; s++) {
        for (int d=0; d<dim; d++) {
            total_heat_flux[d] += species_enthalpy[s] * species_diffusion_flux[s][d];
        }
    }

    return total_heat_flux;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::get_tensor_magnitude_sqr (
    const dealii::Tensor<2,dim,real> &tensor) const
{
    real tensor_magnitude_sqr = 0.0;
    for (int i=0; i<dim; ++i) {
        for (int j=0; j<dim; ++j) {
            tensor_magnitude_sqr += tensor[i][j]*tensor[i][j];
        }
    }
    return tensor_magnitude_sqr;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<2,dim,real> NavierStokes_RealGas<dim,nspecies,nstate,real>
::extract_velocities_gradient_from_primitive_solution_gradient (
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    dealii::Tensor<2,dim,real> velocities_gradient;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            velocities_gradient[d1][d2] = primitive_soln_gradient[1+d1][d2];
        }
    }
    return velocities_gradient;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<2,dim,real> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_strain_rate_tensor (
    const dealii::Tensor<2,dim,real> &vel_gradient) const
{ 
    // Strain rate tensor, S_{i,j}
    dealii::Tensor<2,dim,real> strain_rate_tensor;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            // rate of strain (deformation) tensor:
            strain_rate_tensor[d1][d2] = 0.5*(vel_gradient[d1][d2] + vel_gradient[d2][d1]);
        }
    }
    return strain_rate_tensor;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_strain_rate_tensor_magnitude_sqr (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Get velocity gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = convert_conservative_gradient_to_primitive_gradient(conservative_soln, conservative_soln_gradient);
    const dealii::Tensor<2,dim,real> velocities_gradient = extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);

    // Compute the strain rate tensor
    const dealii::Tensor<2,dim,real> strain_rate_tensor = compute_strain_rate_tensor(velocities_gradient);
    // Get magnitude squared
    real strain_rate_tensor_magnitude_sqr = get_tensor_magnitude_sqr(strain_rate_tensor);
    
    return strain_rate_tensor_magnitude_sqr;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<2,dim,real> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor (
    const real scaled_viscosity_coefficient,
    const dealii::Tensor<2,dim,real> &strain_rate_tensor) const
{
    /* Nondimensionalized viscous stress tensor, $\bm{\tau}^{*}$ 
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */

    // Divergence of velocity
    // -- Initialize
    real vel_divergence; // complex initializes it as 0+0i
    if(std::is_same<real,real>::value){ 
        vel_divergence = 0.0;
    }
    // -- Obtain from trace of strain rate tensor
    for (int d=0; d<dim; d++) {
        vel_divergence += strain_rate_tensor[d][d];
    }

    // Viscous stress tensor, \tau_{i,j}
    dealii::Tensor<2,dim,real> viscous_stress_tensor;
    const real scaled_2nd_viscosity_coefficient = (-2.0/3.0)*scaled_viscosity_coefficient; // Stokes' hypothesis
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            viscous_stress_tensor[d1][d2] = 2.0*scaled_viscosity_coefficient*strain_rate_tensor[d1][d2];
        }
        viscous_stress_tensor[d1][d1] += scaled_2nd_viscosity_coefficient*vel_divergence;
    }
    return viscous_stress_tensor;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<2,dim,real> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_viscous_stress_tensor (
    const std::array<real,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const
{
    /* Nondimensionalized viscous stress tensor, $\bm{\tau}^{*}$ 
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    const dealii::Tensor<2,dim,real> vel_gradient = extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    const dealii::Tensor<2,dim,real> strain_rate_tensor = compute_strain_rate_tensor(vel_gradient);
    const real scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient(primitive_soln);

    // Viscous stress tensor, \tau_{i,j}
    const dealii::Tensor<2,dim,real> viscous_stress_tensor 
        = compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor(scaled_viscosity_coefficient,strain_rate_tensor);

    return viscous_stress_tensor;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokes_RealGas<dim,nspecies,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    /* Nondimensionalized viscous flux (i.e. dissipative flux)
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
     */
    std::array<dealii::Tensor<1,dim,real>,nstate> viscous_flux = dissipative_flux_templated(conservative_soln, solution_gradient);
    return viscous_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokes_RealGas<dim,nspecies,nstate,real>
::dissipative_flux_templated (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    /* Nondimensionalized viscous flux (i.e. dissipative flux)
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
     */

    // Step 1: Primitive solution
    const std::array<real,nstate> primitive_soln = this->template convert_conservative_to_primitive(conservative_soln); // from Real Gas
    
    // Step 2: Gradient of primitive solution
    const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = convert_conservative_gradient_to_primitive_gradient(conservative_soln, solution_gradient);
    
    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const dealii::Tensor<2,dim,real> viscous_stress_tensor = compute_viscous_stress_tensor(primitive_soln, primitive_soln_gradient);
    const dealii::Tensor<1,dim,real> vel = this->template extract_velocities_from_primitive(primitive_soln); // from Real Gas
    const std::array<dealii::Tensor<1,dim,real>, nspecies> species_diffusion_flux = compute_species_diffusion_flux(primitive_soln, primitive_soln_gradient);
    const dealii::Tensor<1,dim,real> total_heat_flux = compute_total_heat_flux(conservative_soln, solution_gradient, species_diffusion_flux);

    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
    const std::array<dealii::Tensor<1,dim,real>,nstate> viscous_flux = dissipative_flux_given_velocities_viscous_stress_tensor_heat_flux_species_diffusion_flux(vel,viscous_stress_tensor,total_heat_flux,species_diffusion_flux);
    return viscous_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokes_RealGas<dim,nspecies,nstate,real>
::dissipative_flux_given_velocities_viscous_stress_tensor_heat_flux_species_diffusion_flux (
    const dealii::Tensor<1,dim,real> &vel,
    const dealii::Tensor<2,dim,real> &viscous_stress_tensor,
    const dealii::Tensor<1,dim,real> &total_heat_flux,
    const std::array<dealii::Tensor<1,dim,real>, nspecies> &species_diffusion_flux) const
{
    /* Nondimensionalized viscous flux (i.e. dissipative flux)
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
     */

    /* Construct viscous flux given velocities, viscous stress tensor,
     * and heat flux; Note: sign corresponds to LHS
     */
    std::array<dealii::Tensor<1,dim,real>,nstate> viscous_flux;
    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        // Mixture Density equation
        viscous_flux[0][flux_dim] = 0.0;
        // Mixture Momentum equation
        for (int stress_dim=0; stress_dim<dim; ++stress_dim){
            viscous_flux[1+stress_dim][flux_dim] = -viscous_stress_tensor[stress_dim][flux_dim];
        }
        // Mixture Energy equation
        viscous_flux[dim+1][flux_dim] = 0.0;
        for (int stress_dim=0; stress_dim<dim; ++stress_dim){
           viscous_flux[dim+1][flux_dim] -= vel[stress_dim]*viscous_stress_tensor[flux_dim][stress_dim];
        }
        viscous_flux[dim+1][flux_dim] += total_heat_flux[flux_dim];
        // Species density equation
        for (int s=0; s<nspecies-1; ++s) {
            viscous_flux[dim+2+s][flux_dim] = species_diffusion_flux[s][flux_dim];
        }
    }
    return viscous_flux;
}

template <int dim, int nspecies, int nstate, typename real>
void NavierStokes_RealGas<dim,nspecies,nstate,real>
::boundary_wall (
   const dealii::Tensor<1,dim,real> &/*normal_int*/,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;

    // No-slip wall boundary conditions
    // Given by equations 460-461 of the following paper
    // Hartmann, Ralf. "Numerical analysis of higher order discontinuous Galerkin finite element methods." (2008): 1-107.
    const std::array<real,nstate> primitive_interior_values = this->template convert_conservative_to_primitive(soln_int);

    // Copy density
    std::array<real,nstate> primitive_boundary_values;
    primitive_boundary_values[0] = primitive_interior_values[0];

    // Associated thermal boundary condition
    if(thermal_boundary_condition_type == thermal_boundary_condition_enum::isothermal) { 
        // isothermal boundary
        primitive_boundary_values[dim+1] = this->compute_pressure_from_density_temperature(primitive_boundary_values[0], isothermal_wall_temperature,soln_int);
    } else if(thermal_boundary_condition_type == thermal_boundary_condition_enum::adiabatic) {
        // adiabatic boundary
        primitive_boundary_values[dim+1] = primitive_interior_values[dim+1];
    }
    
    // No-slip boundary condition on velocity
    dealii::Tensor<1,dim,real> velocities_bc;
    for (int d=0; d<dim; d++) {
        velocities_bc[d] = 0.0;
    }
    for (int d=0; d<dim; ++d) {
        primitive_boundary_values[1+d] = velocities_bc[d];
    }

    // Apply boundary conditions:
    // -- solution at boundary
    const std::array<real,nstate> modified_conservative_boundary_values = this->convert_primitive_to_conservative(primitive_boundary_values);
    for (int istate=0; istate<nstate; ++istate) {
        soln_bc[istate] = modified_conservative_boundary_values[istate];
    }
    // -- gradient of solution at boundary
    for (int istate=0; istate<nstate; ++istate) {
        soln_grad_bc[istate] = soln_grad_int[istate];
    }
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<1,3,real> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_vorticity (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Compute the vorticity
    dealii::Tensor<1,3,real> vorticity;
    for(int d=0; d<3; ++d) {
        vorticity[d] = 0.0;
    }
    if constexpr(dim>1) {
        // Get velocity gradient
        const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = convert_conservative_gradient_to_primitive_gradient(conservative_soln, conservative_soln_gradient);
        const dealii::Tensor<2,dim,real> velocities_gradient = extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
        if constexpr(dim==2) {
            // vorticity exists only in z-component
            vorticity[2] = velocities_gradient[1][0] - velocities_gradient[0][1]; // z-component
        }
        if constexpr(dim==3) {
            vorticity[0] = velocities_gradient[2][1] - velocities_gradient[1][2]; // x-component
            vorticity[1] = velocities_gradient[0][2] - velocities_gradient[2][0]; // y-component
            vorticity[2] = velocities_gradient[1][0] - velocities_gradient[0][1]; // z-component
        }
    }
    return vorticity;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_vorticity_magnitude_sqr (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Compute the vorticity
    dealii::Tensor<1,3,real> vorticity = compute_vorticity(conservative_soln, conservative_soln_gradient);
    // Compute vorticity magnitude squared
    real vorticity_magnitude_sqr = 0.0;
    for(int d=0; d<3; ++d) {
        vorticity_magnitude_sqr += vorticity[d]*vorticity[d];
    }
    return vorticity_magnitude_sqr;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_vorticity_magnitude (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    real vorticity_magnitude_sqr = compute_vorticity_magnitude_sqr(conservative_soln, conservative_soln_gradient);
    real vorticity_magnitude = sqrt(vorticity_magnitude_sqr); 
    return vorticity_magnitude;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_vorticity_based_dissipation_rate_from_integrated_enstrophy (
    const real integrated_enstrophy) const
{
    real dissipation_rate = 2.0*integrated_enstrophy/(this->reynolds_number_inf);
    return dissipation_rate;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_enstrophy (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Compute enstrophy
    const real density = conservative_soln[0];
    real enstrophy = 0.5*density*compute_vorticity_magnitude_sqr(conservative_soln, conservative_soln_gradient);
    return enstrophy;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_incompressible_enstrophy (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Compute incompressible enstrophy
    real enstrophy = 0.5*compute_vorticity_magnitude_sqr(conservative_soln, conservative_soln_gradient);
    return enstrophy;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_pressure_dilatation (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Get pressure
    const real pressure = this->template compute_mixture_pressure(conservative_soln);

    // Compute the pressure dilatation
    real pressure_dilatation = compute_dilatation(conservative_soln,conservative_soln_gradient);
    pressure_dilatation *= pressure;

    return pressure_dilatation;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_dilatation (
    const std::array<real,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const
{
    // Get velocity gradient
    const std::array<dealii::Tensor<1,dim,real>,nstate> primitive_soln_gradient = convert_conservative_gradient_to_primitive_gradient(conservative_soln, conservative_soln_gradient);
    const dealii::Tensor<2,dim,real> velocities_gradient = extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);

    // Compute the dilatation
    real dilatation = 0.0;
    for(int d=0; d<dim; ++d) {
        dilatation += velocities_gradient[d][d]; // divergence
    }

    return dilatation;
}

template <int dim, int nspecies, int nstate, typename real>
real NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_incompressible_palinstrophy (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,3> &vorticity_gradient) const
{
    // Compute vorticity gradient magnitude squared
    real vorticity_gradient_magnitude_sqr = 0.0;
    for(int istate=0; istate<3; ++istate) {
        for(int d=0; d<dim; ++d) {
            vorticity_gradient_magnitude_sqr += vorticity_gradient[istate][d]*vorticity_gradient[istate][d];
        }
    }
    // Compute incompressible palinstrophy
    const real palinstrophy = 0.5*vorticity_gradient_magnitude_sqr;
    return palinstrophy;
}


template <int dim, int nspecies, int nstate, typename real>
dealii::Vector<double> NavierStokes_RealGas<dim,nspecies,nstate,real>::post_compute_derived_quantities_vector (
    const dealii::Vector<double>              &uh,
    const std::vector<dealii::Tensor<1,dim> > &duh,
    const std::vector<dealii::Tensor<2,dim> > &dduh,
    const dealii::Tensor<1,dim>               &normals,
    const dealii::Point<dim>                  &evaluation_points) const
{
    std::vector<std::string> names = post_get_names ();
    dealii::Vector<double> computed_quantities = PhysicsBase<dim,nspecies,nstate,real>::post_compute_derived_quantities_vector ( uh, duh, dduh, normals, evaluation_points);
    unsigned int current_data_index = computed_quantities.size() - 1;
    computed_quantities.grow_or_shrink(names.size());
    if constexpr (std::is_same<real,double>::value) {
        // get the solution
        std::array<double, nstate> conservative_soln;
        for (unsigned int s=0; s<nstate; ++s) {
            conservative_soln[s] = uh(s);
        }
    
        // get the solution gradient
        std::array<dealii::Tensor<1,dim,double>,nstate> conservative_soln_gradient;
        for (unsigned int s=0; s<nstate; ++s) {
            for (unsigned int d=0; d<dim; ++d) {
                conservative_soln_gradient[s][d] = duh[s][d];
            }
        }
        // Mixture density
        computed_quantities(++current_data_index) = this->template compute_mixture_density(conservative_soln);
        // Velocities
        const dealii::Tensor<1,dim,real> vel = this->template compute_velocities(conservative_soln);
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = vel[d];
        }
        // Mixture momentum
        for (unsigned int d=0; d<dim; ++d) {
            computed_quantities(++current_data_index) = conservative_soln[1+d];
        }
        // Mixture energy
        computed_quantities(++current_data_index) = this->template compute_mixture_specific_total_energy(conservative_soln);
        // Mixture pressure
        computed_quantities(++current_data_index) = this->template compute_mixture_pressure(conservative_soln);
        // Non-dimensional temperature
        computed_quantities(++current_data_index) = this->template compute_temperature(conservative_soln); 
        // Dimensional temperature
        computed_quantities(++current_data_index) = this->template compute_dimensional_temperature(this->template compute_temperature(conservative_soln));
        // Mixture specific total enthalpy
        computed_quantities(++current_data_index) = this->template compute_mixture_specific_total_enthalpy(conservative_soln);  
        // Mass fractions
        const std::array<real,nspecies> mass_fractions = this->template compute_mass_fractions(conservative_soln);
        for (unsigned int s=0; s<nspecies; ++s) 
        {
            computed_quantities(++current_data_index) = mass_fractions[s];
        }
        // Species densities
        const std::array<real,nspecies> species_densities = this->template compute_species_densities(conservative_soln);
        for (unsigned int s=0; s<nspecies; ++s) 
        {
            computed_quantities(++current_data_index) = species_densities[s];
        }
        // Vorticity
        dealii::Tensor<1,3,double> vorticity = compute_vorticity(conservative_soln,conservative_soln_gradient);
        for (unsigned int d=0; d<3; ++d) {
            computed_quantities(++current_data_index) = vorticity[d];
        }
    }
    if (computed_quantities.size()-1 != current_data_index) {
        std::cout << " Did not assign a value to all the data. Missing " << computed_quantities.size() - current_data_index << " variables."
                  << " If you added a new output variable, make sure the names and DataComponentInterpretation match the above. "
                  << std::endl;
    }

    return computed_quantities;
}

template <int dim, int nspecies, int nstate, typename real>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> NavierStokes_RealGas<dim,nspecies,nstate,real>
::post_get_data_component_interpretation () const
{
    namespace DCI = dealii::DataComponentInterpretation;
    std::vector<DCI::DataComponentInterpretation> interpretation = PhysicsBase<dim,nspecies,nstate,real>::post_get_data_component_interpretation (); // state variables
    interpretation.push_back (DCI::component_is_scalar); // Mixture density
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Velocity
    }
    for (unsigned int d=0; d<dim; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // Mixture momentum
    }
    interpretation.push_back (DCI::component_is_scalar); // Mixture energy
    interpretation.push_back (DCI::component_is_scalar); // Mixture pressure
    interpretation.push_back (DCI::component_is_scalar); // Non-dimensional temperature
    interpretation.push_back (DCI::component_is_scalar); // Dimensional temperature
    interpretation.push_back (DCI::component_is_scalar); // Mixture specific total enthalpy
    for (unsigned int s=0; s<nspecies; ++s) {
         interpretation.push_back (DCI::component_is_scalar); // Mass fractions
    }
    for (unsigned int s=0; s<nspecies; ++s) {
        interpretation.push_back (DCI::component_is_scalar); // Species densities
    }
    for (unsigned int d=0; d<3; ++d) {
        interpretation.push_back (DCI::component_is_part_of_vector); // vorticity
    }

    std::vector<std::string> names = post_get_names();
    if (names.size() != interpretation.size()) {
        std::cout << "Number of DataComponentInterpretation is not the same as number of names for output file" << std::endl;
    }
    return interpretation;
}

template <int dim, int nspecies, int nstate, typename real>
std::vector<std::string> NavierStokes_RealGas<dim,nspecies,nstate,real>
::post_get_names () const
{
    std::vector<std::string> names = PhysicsBase<dim,nspecies,nstate,real>::post_get_names ();
    names.push_back ("mixture_density");
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("velocity");
    }
    for (unsigned int d=0; d<dim; ++d) {
      names.push_back ("mixture_momentum");
    }
    names.push_back ("mixture_energy");
    names.push_back ("mixture_pressure");
    names.push_back ("temperature");
    names.push_back ("dimensional_temperature");
    names.push_back ("mixture_specific_total_enthalpy");
    for (unsigned int s=0; s<nspecies; ++s) 
    {
      std::string string_mass_fraction = "mass_fraction";
      std::string string_species_mass_fraction = string_mass_fraction + "_" + this->species_name[s];
      names.push_back (string_species_mass_fraction);
    }
    for (unsigned int s=0; s<nspecies; ++s) 
    {
      std::string string_density = "species_density";
      std::string string_species_density = string_density + "_" + this->species_name[s];
      names.push_back (string_species_density);
    }
    for (unsigned int d=0; d<3; ++d) {
        names.push_back ("vorticity");
    }

    return names;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::UpdateFlags NavierStokes_RealGas<dim,nspecies,nstate,real>
::post_get_needed_update_flags () const
{
    //return update_values | update_gradients;
    return dealii::update_values 
            | dealii::update_gradients
            | dealii::update_quadrature_points;
}

template class NavierStokes_RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, double >;
template class NavierStokes_RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, FadType >;
template class NavierStokes_RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, RadType >;
template class NavierStokes_RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, FadFadType >;
template class NavierStokes_RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, RadFadType >;
} // Physics namespace
} // PHiLiP namespace

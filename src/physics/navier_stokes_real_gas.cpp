#include <cmath>
#include <vector>
#include <complex> // for the jacobian
#include <boost/preprocessor/seq/for_each.hpp>

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
std::array<dealii::Tensor<1,dim,double>,nstate> NavierStokes_RealGas<dim,nspecies,nstate,real>
::convert_conservative_gradient_to_primitive_gradient (
    const std::array<double,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,double>,nstate> &conservative_soln_gradient) const
{
    // conservative_soln_gradient is solution_gradient
    std::array<dealii::Tensor<1,dim,double>,nstate> primitive_soln_gradient;

    // get primitive solution
    const std::array<double,nstate> primitive_soln = this->template convert_conservative_to_primitive(conservative_soln); // from Euler

    // extract from primitive solution
    const double density = primitive_soln[0];
    const dealii::Tensor<1,dim,double> vel = this->template extract_velocities_from_primitive(primitive_soln); // from Euler

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

//     // pressure gradient
//     // -- formulation 1:
//     // const double vel2 = this->template compute_velocity_squared(vel); // from Euler
//     // for (int d1=0; d1<dim; d1++) {
//     //     primitive_soln_gradient[nstate-1][d1] = conservative_soln_gradient[nstate-1][d1] - 0.5*vel2*conservative_soln_gradient[0][d1];
//     //     for (int d2=0; d2<dim; d2++) {
//     //         primitive_soln_gradient[nstate-1][d1] -= conservative_soln[1+d2]*primitive_soln_gradient[1+d2][d1];
//     //     }
//     //     primitive_soln_gradient[nstate-1][d1] *= this->gamm1;
//     // }
//     // -- formulation 2 (equivalent to formulation 1):
    for (int d1=0; d1<dim; d1++) {
        primitive_soln_gradient[dim+1][d1] = conservative_soln_gradient[dim+1][d1];
        for (int d2=0; d2<dim; d2++) {
            primitive_soln_gradient[dim+1][d1] -= 0.5*(primitive_soln[1+d2]*conservative_soln_gradient[1+d2][d1]  
                                                           + conservative_soln[1+d2]*primitive_soln_gradient[1+d2][d1]);
        }
        primitive_soln_gradient[dim+1][d1] *= this->gamm1;
    }
    return primitive_soln_gradient;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<1,dim,double> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_temperature_gradient (
    const std::array<double,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,double>,nstate> &primitive_soln_gradient) const
{
    const double density = primitive_soln[0];
    const double temperature = this->template compute_temperature(primitive_soln); // from Euler

    dealii::Tensor<1,dim,double> temperature_gradient;
    for (int d=0; d<dim; d++) {
        temperature_gradient[d] = (this->gam_ref*this->mach_ref_sqr*primitive_soln_gradient[dim+1][d] - temperature*primitive_soln_gradient[0][d])/density;
    }
    return temperature_gradient;
}

template <int dim, int nspecies, int nstate, typename real>
inline double NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_viscosity_coefficient (const std::array<double,nstate> &primitive_soln) const
{   
    // Use either Sutherland's law or constant viscosity
    double viscosity_coefficient;
    if(use_constant_viscosity){
        viscosity_coefficient = 1.0*constant_viscosity;
    } else {
        viscosity_coefficient = compute_viscosity_coefficient_sutherlands_law(primitive_soln);
    }

    return viscosity_coefficient;
}

template <int dim, int nspecies, int nstate, typename real>
inline double NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_viscosity_coefficient_sutherlands_law (const std::array<double,nstate> &primitive_soln) const
{
    /* Nondimensionalized viscosity coefficient, \mu^{*}
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     * Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    const double temperature = this->template compute_temperature(primitive_soln); // from Euler

    const double viscosity_coefficient = ((1.0 + temperature_ratio)/(temperature + temperature_ratio))*pow(temperature,1.5);
    
    return viscosity_coefficient;
}

template <int dim, int nspecies, int nstate, typename real>
inline double NavierStokes_RealGas<dim,nspecies,nstate,real>
::scale_viscosity_coefficient (const double viscosity_coefficient) const
{
    /* Scaled nondimensionalized viscosity coefficient, $\hat{\mu}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    const double scaled_viscosity_coefficient = viscosity_coefficient/reynolds_number_inf;
    
    return scaled_viscosity_coefficient;
}

template <int dim, int nspecies, int nstate, typename real>
inline double NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_scaled_viscosity_coefficient (const std::array<double,nstate> &primitive_soln) const
{
    /* Scaled nondimensionalized viscosity coefficient, $\hat{\mu}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    const double viscosity_coefficient = compute_viscosity_coefficient(primitive_soln);
    const double scaled_viscosity_coefficient = scale_viscosity_coefficient(viscosity_coefficient);

    return scaled_viscosity_coefficient;
}

template <int dim, int nspecies, int nstate, typename real>
inline double NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number (const double scaled_viscosity_coefficient, const double prandtl_number_input) const
{
    /* Scaled nondimensionalized heat conductivity, $\hat{\kappa}^{*}$, given the scaled viscosity coefficient
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    const double scaled_heat_conductivity = scaled_viscosity_coefficient/(this->gamm1*this->mach_ref_sqr*prandtl_number_input);
    
    return scaled_heat_conductivity;
}

template <int dim, int nspecies, int nstate, typename real>
inline double NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_scaled_heat_conductivity (const std::array<double,nstate> &primitive_soln) const
{
    /* Scaled nondimensionalized heat conductivity, $\hat{\kappa}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    const double scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient(primitive_soln);

    const double scaled_heat_conductivity = compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number(scaled_viscosity_coefficient,prandtl_number);
    
    return scaled_heat_conductivity;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<1,dim,double> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_heat_flux (
    const std::array<double,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,double>,nstate> &primitive_soln_gradient) const
{
    /* Nondimensionalized heat flux, $\bm{q}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    const double scaled_heat_conductivity = compute_scaled_heat_conductivity(primitive_soln);
    const dealii::Tensor<1,dim,double> temperature_gradient = compute_temperature_gradient(primitive_soln, primitive_soln_gradient);
    // Compute the heat flux
    const dealii::Tensor<1,dim,double> heat_flux = compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient(scaled_heat_conductivity,temperature_gradient);
    return heat_flux;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<1,dim,double> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient (
    const double scaled_heat_conductivity,
    const dealii::Tensor<1,dim,double> &temperature_gradient) const
{
    /* Nondimensionalized heat flux, $\bm{q}^{*}$
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    dealii::Tensor<1,dim,double> heat_flux;
    for (int d=0; d<dim; d++) {
        heat_flux[d] = -scaled_heat_conductivity*temperature_gradient[d];
    }
    return heat_flux;
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
dealii::Tensor<2,dim,double> NavierStokes_RealGas<dim,nspecies,nstate,real>
::extract_velocities_gradient_from_primitive_solution_gradient (
    const std::array<dealii::Tensor<1,dim,double>,nstate> &primitive_soln_gradient) const
{
    dealii::Tensor<2,dim,double> velocities_gradient;
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            velocities_gradient[d1][d2] = primitive_soln_gradient[1+d1][d2];
        }
    }
    return velocities_gradient;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<2,dim,double> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_strain_rate_tensor (
    const dealii::Tensor<2,dim,double> &vel_gradient) const
{ 
    // Strain rate tensor, S_{i,j}
    dealii::Tensor<2,dim,double> strain_rate_tensor;
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
dealii::Tensor<2,dim,double> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor (
    const double scaled_viscosity_coefficient,
    const dealii::Tensor<2,dim,double> &strain_rate_tensor) const
{
    /* Nondimensionalized viscous stress tensor, $\bm{\tau}^{*}$ 
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */

    // Divergence of velocity
    // -- Initialize
    double vel_divergence; // complex initializes it as 0+0i
    if(std::is_same<double,real>::value){ 
        vel_divergence = 0.0;
    }
    // -- Obtain from trace of strain rate tensor
    for (int d=0; d<dim; d++) {
        vel_divergence += strain_rate_tensor[d][d];
    }

    // Viscous stress tensor, \tau_{i,j}
    dealii::Tensor<2,dim,double> viscous_stress_tensor;
    const double scaled_2nd_viscosity_coefficient = (-2.0/3.0)*scaled_viscosity_coefficient; // Stokes' hypothesis
    for (int d1=0; d1<dim; d1++) {
        for (int d2=0; d2<dim; d2++) {
            viscous_stress_tensor[d1][d2] = 2.0*scaled_viscosity_coefficient*strain_rate_tensor[d1][d2];
        }
        viscous_stress_tensor[d1][d1] += scaled_2nd_viscosity_coefficient*vel_divergence;
    }
    return viscous_stress_tensor;
}

template <int dim, int nspecies, int nstate, typename real>
dealii::Tensor<2,dim,double> NavierStokes_RealGas<dim,nspecies,nstate,real>
::compute_viscous_stress_tensor (
    const std::array<double,nstate> &primitive_soln,
    const std::array<dealii::Tensor<1,dim,double>,nstate> &primitive_soln_gradient) const
{
    /* Nondimensionalized viscous stress tensor, $\bm{\tau}^{*}$ 
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    const dealii::Tensor<2,dim,double> vel_gradient = extract_velocities_gradient_from_primitive_solution_gradient(primitive_soln_gradient);
    const dealii::Tensor<2,dim,double> strain_rate_tensor = compute_strain_rate_tensor(vel_gradient);
    const double scaled_viscosity_coefficient = compute_scaled_viscosity_coefficient(primitive_soln);

    // Viscous stress tensor, \tau_{i,j}
    const dealii::Tensor<2,dim,double> viscous_stress_tensor 
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
std::array<dealii::Tensor<1,dim,double>,nstate> NavierStokes_RealGas<dim,nspecies,nstate,real>
::dissipative_flux_templated (
    const std::array<double,nstate> &conservative_soln,
    const std::array<dealii::Tensor<1,dim,double>,nstate> &solution_gradient) const
{
    /* Nondimensionalized viscous flux (i.e. dissipative flux)
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
     */

    // Step 1: Primitive solution
    const std::array<double,nstate> primitive_soln = this->template convert_conservative_to_primitive(conservative_soln); // from Euler
    
    // Step 2: Gradient of primitive solution
    const std::array<dealii::Tensor<1,dim,double>,nstate> primitive_soln_gradient = convert_conservative_gradient_to_primitive_gradient(conservative_soln, solution_gradient);
    
    // Step 3: Viscous stress tensor, Velocities, Heat flux
    const dealii::Tensor<2,dim,double> viscous_stress_tensor = compute_viscous_stress_tensor(primitive_soln, primitive_soln_gradient);
    const dealii::Tensor<1,dim,double> vel = this->template extract_velocities_from_primitive(primitive_soln); // from Euler
    const dealii::Tensor<1,dim,double> heat_flux = compute_heat_flux(primitive_soln, primitive_soln_gradient);

    // Step 4: Construct viscous flux; Note: sign corresponds to LHS
    const std::array<dealii::Tensor<1,dim,double>,nstate> viscous_flux = dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux(vel,viscous_stress_tensor,heat_flux);
    return viscous_flux;
}

template <int dim, int nspecies, int nstate, typename real>
std::array<dealii::Tensor<1,dim,double>,nstate> NavierStokes_RealGas<dim,nspecies,nstate,real>
::dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux (
    const dealii::Tensor<1,dim,double> &vel,
    const dealii::Tensor<2,dim,double> &viscous_stress_tensor,
    const dealii::Tensor<1,dim,double> &heat_flux) const
{
    /* Nondimensionalized viscous flux (i.e. dissipative flux)
     * Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.12.1-4.12.4)
     */

    /* Construct viscous flux given velocities, viscous stress tensor,
     * and heat flux; Note: sign corresponds to LHS
     */
    std::array<dealii::Tensor<1,dim,double>,nstate> viscous_flux;
    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        // Density equation
        viscous_flux[0][flux_dim] = 0.0;
        // Momentum equation
        for (int stress_dim=0; stress_dim<dim; ++stress_dim){
            viscous_flux[1+stress_dim][flux_dim] = -viscous_stress_tensor[stress_dim][flux_dim];
        }
        // Energy equation
        viscous_flux[dim+1][flux_dim] = 0.0;
        for (int stress_dim=0; stress_dim<dim; ++stress_dim){
           viscous_flux[dim+1][flux_dim] -= vel[stress_dim]*viscous_stress_tensor[flux_dim][stress_dim];
        }
        viscous_flux[dim+1][flux_dim] += heat_flux[flux_dim];
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



template class NavierStokes_RealGas < PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+PHILIP_SPECIES+1, double >;
} // Physics namespace
} // PHiLiP namespace

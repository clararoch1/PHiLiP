#ifndef __NAVIER_STOKES_REAL_GAS__
#define __NAVIER_STOKES_REAL_GAS__

#include "real_gas.h"
#include "parameters/parameters_navier_stokes.h"

namespace PHiLiP {
namespace Physics {

/// Navier-Stokes equations. Derived from Real gas for the convective terms, which is derived from PhysicsBase. 
template <int dim, int nspecies, int nstate, typename real>
class NavierStokes_RealGas: public RealGas <dim, nspecies, nstate, real>
{
protected:
    // For overloading the virtual functions defined in PhysicsBase
    /** Once you overload a function from Base class in Derived class,
     *  all functions with the same name in the Base class get hidden in Derived class.  
     *  
     *  Solution: In order to make the hidden function visible in derived class, 
     *  we need to add the following:
    */
    using PhysicsBase<dim,nspecies,nstate,real>::dissipative_flux;
    using PhysicsBase<dim,nspecies,nstate,real>::source_term;
public:
    using thermal_boundary_condition_enum = Parameters::NavierStokesParam::ThermalBoundaryCondition;
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    NavierStokes_RealGas( 
        const Parameters::AllParameters *const                    parameters_input,
        const double                                              prandtl_number,
        const double                                              reynolds_number_inf,
        const bool                                                use_constant_viscosity,
        const double                                              constant_viscosity,
        const double                                              temperature_inf = 273.15,
        const double                                              isothermal_wall_temperature = 1.0,
        const thermal_boundary_condition_enum                     thermal_boundary_condition_type = thermal_boundary_condition_enum::adiabatic);

    /// Nondimensionalized viscosity coefficient at infinity.
    const double viscosity_coefficient_inf;
    /// Flag to use constant viscosity instead of Sutherland's law of viscosity
    const bool use_constant_viscosity;
    /// Nondimensionalized constant viscosity
    const double constant_viscosity;
    /// Prandtl number
    const double prandtl_number;
    /// Farfield (free stream) Reynolds number
    const double reynolds_number_inf;
    /// Nondimensionalized isothermal wall temperature
    const double isothermal_wall_temperature;
    /// Thermal boundary condition type (adiabatic or isothermal)
    const thermal_boundary_condition_enum thermal_boundary_condition_type;

protected:    
    ///@{
    /** Constants for Sutherland's law for viscosity
     *  Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     *  Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    const double sutherlands_temperature; ///< Sutherland's temperature. Units: [K]
    const double freestream_temperature; ///< Freestream temperature. Units: [K]
    const double temperature_ratio; ///< Ratio of Sutherland's temperature to freestream temperature
    //@}

public:

    /** Obtain gradient of primitive variables from gradient of conservative variables */
    std::array<dealii::Tensor<1,dim,real>,nstate> 
    convert_conservative_gradient_to_primitive_gradient (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Nondimensionalized temperature gradient */
    dealii::Tensor<1,dim,real> compute_temperature_gradient (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized viscosity coefficient, mu*
     *  Based on the use_constant_viscosity flag, it returns a value based on either:
     *  (1) Sutherland's viscosity law, or
     *  (2) Constant nondimensionalized viscosity value
     */
    real compute_viscosity_coefficient (const std::array<real,nstate> &primitive_soln) const;

    /** Nondimensionalized viscosity coefficient, mu*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.16)
     * 
     *  Based on Sutherland's law for viscosity
     * * Reference: Sutherland, W. (1893), "The viscosity of gases and molecular force", Philosophical Magazine, S. 5, 36, pp. 507-531 (1893)
     * * Values: https://www.cfd-online.com/Wiki/Sutherland%27s_law
     */
    real compute_viscosity_coefficient_sutherlands_law (const std::array<real,nstate> &primitive_soln) const;

    /** Scaled nondimensionalized viscosity coefficient, hat{mu*}, given nondimensionalized viscosity coefficient
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    real scale_viscosity_coefficient (const real viscosity_coefficient) const;

    /** Scaled nondimensionalized viscosity coefficient, hat{mu*} 
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.14)
     */
    real compute_scaled_viscosity_coefficient (const std::array<real,nstate> &primitive_soln) const;

    /** Scaled nondimensionalized heat conductivity, hat{kappa*}, given scaled nondimensionalized viscosity coefficient and Prandtl number
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    real compute_scaled_heat_conductivity_given_scaled_viscosity_coefficient_and_prandtl_number (
        const real scaled_viscosity_coefficient, 
        const real prandtl_number_input) const;

    /** Scaled nondimensionalized heat conductivity, hat{kappa*}
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    real compute_scaled_heat_conductivity (const std::array<real,nstate> &primitive_soln) const;

    /** Nondimensionalized heat flux, q*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    dealii::Tensor<1,dim,real> compute_heat_flux (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized heat flux, q*, given the scaled heat conductivity and temperature gradient
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.13)
     */
    dealii::Tensor<1,dim,real> compute_heat_flux_given_scaled_heat_conductivity_and_temperature_gradient (
        const real scaled_heat_conductivity,
        const dealii::Tensor<1,dim,real> &temperature_gradient) const;

    /** Extract gradient of velocities */
    dealii::Tensor<2,dim,real> extract_velocities_gradient_from_primitive_solution_gradient (
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized strain rate tensor, S*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, extracted from eq.(4.14.12)
     */
    dealii::Tensor<2,dim,real> compute_strain_rate_tensor (
        const dealii::Tensor<2,dim,real> &vel_gradient) const;

    /// Evaluate the square of the strain-rate tensor magnitude (i.e. double dot product) from conservative variables and gradient of conservative variables
    real compute_strain_rate_tensor_magnitude_sqr (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Nondimensionalized viscous stress tensor, tau*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    dealii::Tensor<2,dim,real> compute_viscous_stress_tensor_via_scaled_viscosity_and_strain_rate_tensor (
        const real scaled_viscosity_coefficient,
        const dealii::Tensor<2,dim,real> &strain_rate_tensor) const;

    /** Nondimensionalized viscous stress tensor, tau*
     *  Reference: Masatsuka 2018 "I do like CFD", p.148, eq.(4.14.12)
     */
    dealii::Tensor<2,dim,real> compute_viscous_stress_tensor (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /** Nondimensionalized viscous flux (i.e. dissipative flux)
     *  Reference: Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4)
     */
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const override;

    /** Nondimensionalized viscous flux (i.e. dissipative flux) computed 
     *  via given velocities, viscous stress tensor, and heat flux. 
     *  Reference: Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4)
     */
    std::array<dealii::Tensor<1,dim,real>,nstate> 
    dissipative_flux_given_velocities_viscous_stress_tensor_and_heat_flux (
        const dealii::Tensor<1,dim,real> &vel,
        const dealii::Tensor<2,dim,real> &viscous_stress_tensor,
        const dealii::Tensor<1,dim,real> &heat_flux) const;

protected:

    /** Nondimensionalized viscous flux (i.e. dissipative flux)
     *  Reference: Masatsuka 2018 "I do like CFD", p.142, eq.(4.12.1-4.12.4)
     */
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux_templated (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /** No-slip wall boundary conditions
     *  * Given by equations 460-461 of the following paper:
     *  * * Hartmann, Ralf. "Numerical analysis of higher order discontinuous Galerkin finite element methods." (2008): 1-107.
     */
    void boundary_wall (
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const override;

private:
    /// Returns the square of the magnitude of the tensor (i.e. the double dot product of a tensor with itself)
    real get_tensor_magnitude_sqr (const dealii::Tensor<2,dim,real> &tensor) const;

};

} // Physics namespace
} // PHiLiP namespace

#endif

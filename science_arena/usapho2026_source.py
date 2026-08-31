#!/usr/bin/env python3
"""Human-audited source data for the 2026 USAPhO benchmark.

Only the two local PDFs named below were used. The PDFs contain worked
solutions but no point allocations or grading rubric. All criterion weights
in this file are therefore transparent benchmark-derived weights, never
claimed as official USAPhO grading.
"""

from __future__ import annotations


def criterion(cid: str, points: float, description: str, *pages: int) -> dict:
    return {
        "criterion_id": cid,
        "max_score": points,
        "description": description,
        "source_pdf_pages": list(pages),
    }


REFERENCE = (
    "The official reference sheet supplies g=9.8 N/kg, G=6.67e-11 N m^2/kg^2, "
    "k=1/(4 pi epsilon_0)=8.99e9 N m^2/C^2, mu_0/(4 pi)=1e-7 T m/A, "
    "c=3.00e8 m/s, k_B=1.38e-23 J/K, N_A=6.02e23 mol^-1, "
    "R=N_A k_B=8.31 J/(mol K), sigma_SB=5.67e-8 J/(s m^2 K^4), "
    "e=1.602e-19 C, 1 eV=1.602e-19 J, "
    "h=6.63e-34 J s=4.14e-15 eV s, and "
    "m_e=9.109e-31 kg=0.511 MeV/c^2. It also supplies the approximations "
    "(1+x)^n approximately 1+nx+n(n-1)x^2/2 for |nx|<<1, "
    "exp(x) approximately 1+x+x^2/2+x^3/6 for |x|<<1, "
    "sin(theta) approximately theta-theta^3/6 for |theta|<<1, and "
    "cos(theta) approximately 1-theta^2/2 for |theta|<<1."
)


PROBLEMS = {
    "A1": {
        "title": "404 Vision Not Found",
        "part": "A",
        "stem": (
            "A robot eye has an adjustable thin internal lens, a flat sensor s'=10.0 cm behind it, "
            "and a circular pupil of diameter d=5.00 mm. The internal lens focuses sharply for object "
            "distances 1.00 m <= s <= 10.0 m. A thin corrective lens can be placed immediately in front "
            "of the eye lens, with their separation neglected. Treat all lenses as thin."
        ),
        "visual": (
            "The question page contains a schematic eye: incoming parallel rays travel from right to left "
            "through a corrective lens immediately adjacent to the internal lens, then 10.0 cm to the sensor. "
            "All geometry needed for calculation is also stated in the text; the drawing contains no extra scale data."
        ),
        "has_figure": True,
    },
    "A2": {
        "title": "Straight on a Cone",
        "part": "A",
        "stem": (
            "A point mass m is constrained to the surface of an infinite fixed frictionless cone of half-angle "
            "alpha. Ignore gravity; the only force is the cone's normal force. At t=0 the mass is at A, slant "
            "distance r0 from vertex O, with speed v perpendicular to OA. Later it is at B, distance r1 from O."
        ),
        "visual": "No problem-specific diagram is printed; the official page contains only text and equations.",
        "has_figure": False,
    },
    "A3": {
        "title": "Storm Chaser",
        "part": "A",
        "stem": (
            "Neglect moisture and model the atmosphere as an ideal gas of molar mass mu, with molar heat "
            "capacities C_P=C_V+R and gravitational acceleration g. The problem first treats an adiabatically "
            "displaced parcel in a vertically stratified atmosphere, then an isothermal uniform-density Rankine "
            "vortex with v=Omega r for r<R and v=Omega R^2/r for r>R."
        ),
        "visual": "No problem-specific diagram is printed; the official page contains only text and equations.",
        "has_figure": False,
    },
    "B1": {
        "title": "Radiant Bubbles",
        "part": "B",
        "stem": (
            "A star in uniform hydrogen emits ionizing photons at rate Q_* and makes a fully ionized Stromgren "
            "sphere. The recombination rate density is R=n_p n_e alpha. Later parts treat an early ionization "
            "front, pressure-driven expansion between T_in about 10^4 K and T_out about 100 K, and isothermal "
            "hot gas in a galaxy-cluster potential with dPhi/dr=v_c^2/r and free-free emissivity proportional "
            "to n_e^2 sqrt(T)."
        ),
        "visual": "No problem-specific diagram is printed; the official pages contain only text and equations.",
        "has_figure": False,
    },
    "B2": {
        "title": "Say Cheese... It's a Conic",
        "part": "B",
        "stem": (
            "A fixed +1 C charge alpha is at the origin. Attractive orbits use r=r0/(1+e cos phi), e<1; "
            "repulsive orbits use r=r0/(e cos phi-1), e>1. Three positions are photographed at equal time "
            "intervals. In the scattering family of part (c), identical +2 C particles have common speed v_inf "
            "and parallel incoming asymptotic velocity directions; impact parameter B is the incoming-line "
            "distance to the origin, and alpha_s is the angle between incoming and outgoing velocity directions."
        ),
        "visual": "No graph or orbit drawing is printed; coordinates and conic equations are supplied as text.",
        "has_figure": False,
    },
    "B3": {
        "title": "The Trojan Cylinder",
        "part": "B",
        "stem": (
            "The quality factor is Q=2 pi times maximum stored energy divided by energy lost per cycle. A long "
            "superconducting solenoid of inductance L is tightly wound around a thin conducting cylindrical shell "
            "of radius a, thickness t<<a, conductivity sigma, and length ell. End and skin effects are ignored, "
            "and the solenoid current is I(t)=I0 cos(omega t)."
        ),
        "visual": "No circuit or apparatus diagram is printed; all geometry and definitions are in text.",
        "has_figure": False,
    },
}


def staticization_spec(requirements: str, supplemental_context: str = "") -> dict:
    """Question-paper-only completeness record for one atomic solver turn."""
    return {
        "requirements": requirements,
        "supplemental_context": supplemental_context,
    }


# This is deliberately exhaustive rather than a sparse list of known defects.
# The builder fails closed unless these 34 keys exactly match ITEMS.  Every
# supplemental string is transcribed/normalized only from the Problems PDF;
# no solution, reference answer, rubric, or adjudication is used here.
STATICIZATION_SPECS = {
    "A1:a": staticization_spec(
        "Eye geometry s'=10.0 cm and d=5.00 mm; adjustable object-distance interval 1.00-10.0 m; thin-lens assumption; request for both f and P ranges."
    ),
    "A1:b": staticization_spec(
        "Eye focusing interval; stars at infinity; corrective lens placement and negligible separation; requested corrective optical-power range.",
        "The robot is being adapted to observe stars, which are objects at infinity. The thin corrective lens is placed immediately in front of the eye lens, and their separation is neglected.",
    ),
    "A1:c": staticization_spec(
        "Eye focusing interval; the two new functions (stars and reading at 25.0 cm); corrective-lens placement; single-lens feasibility request.",
        "The two proposed functions are observing stars (objects at infinity) and reading printed material at 25.0 cm. A single thin corrective lens would be placed immediately in front of the eye lens, with their separation neglected.",
    ),
    "A1:d": staticization_spec(
        "Pupil diameter 5.00 mm; viewing distance 25.0 cm; wavelength 550 nm; diffraction-limited resolution requested in dpi."
    ),
    "A1:e_i": staticization_spec(
        "Stellar object; image focused on sensor; P0 is total optical power collected by pupil; diffraction makes a central Airy spot; intensity request.",
        "For part (e), while viewing a stellar object, the eye lens is adjusted so that the image forms on the sensor. The total optical power collected by the pupil is P0. Diffraction at the pupil makes the image a central bright spot (Airy disk).",
    ),
    "A1:e_ii": staticization_spec(
        "Complete part-(e) stellar/Airy/P0 setup; same object; pupil changes from 5.0 mm to 2.0 mm; intensity-factor request.",
        "For part (e), while viewing a stellar object, the eye lens is adjusted so that the image forms on the sensor. The total optical power collected by the pupil is P0. Diffraction at the pupil makes the image a central bright spot (Airy disk). The object is unchanged when the pupil diameter is reduced.",
    ),
    "A2:a": staticization_spec(
        "Infinite fixed frictionless cone; half-angle alpha; no gravity; only normal force; initial position r0 and transverse speed v; normal force requested."
    ),
    "A2:b": staticization_spec(
        "Complete cone and initial conditions; later point B at slant distance r1; angle between velocity and OB requested."
    ),
    "A2:c": staticization_spec(
        "Complete cone and initial conditions; later point B at slant distance r1; travel time requested."
    ),
    "A2:d": staticization_spec(
        "Complete cone and initial conditions; long-time total revolutions about cone axis requested."
    ),
    "A3:updraft_a": staticization_spec(
        "Ideal atmosphere, molar mass mu, CP=CV+R, gravity g; pressure-altitude gradient in terms of rho requested."
    ),
    "A3:updraft_b": staticization_spec(
        "Rapidly lofted parcel; no heat exchange; pressure equilibrium with environment; use part (a) and first law; Gamma_p definition.",
        "A parcel of air is lofted rapidly upward in the storm's updraft and exchanges no heat with its surroundings. It remains in pressure equilibrium with the surrounding atmosphere as it rises. Use the pressure-altitude result requested in part (a) together with the first law of thermodynamics. Define the parcel lapse rate as Gamma_p=-dT_p/dz.",
    ),
    "A3:updraft_c_i": staticization_spec(
        "Environmental T_env(z), Gamma_env definition, T_env(0)=T0, initial parcel equilibrium, upward displacement delta z, and Gamma_p definition; instability condition requested.",
        "The environment has temperature profile T_env(z), lapse rate Gamma_env=-dT_env/dz, and surface temperature T_env(0)=T0. A parcel initially in equilibrium at z=0 is displaced upward by delta z. Part (b) defines the parcel lapse rate Gamma_p=-dT_p/dz.",
    ),
    "A3:updraft_c_ii": staticization_spec(
        "Environmental T_env(z), Gamma_env definition, T0, initial equilibrium, displacement delta z, Gamma_p definition, and stable-regime frequency request.",
        "The environment has temperature profile T_env(z), lapse rate Gamma_env=-dT_env/dz, and surface temperature T_env(0)=T0. A parcel initially in equilibrium at z=0 is displaced upward by delta z. Part (b) defines the parcel lapse rate Gamma_p=-dT_p/dz.",
    ),
    "A3:vortex_a": staticization_spec(
        "Isothermal uniform rho0; ambient P_infinity; both Rankine velocity branches; boundary P->P_infinity; small-parcel radial balance; Delta P definition.",
        "For Part B, treat the atmosphere as isothermal with uniform density rho0 and ambient pressure P_infinity. The Rankine vortex has v=Omega r for r<R and v=Omega R^2/r for r>R, with P approaching P_infinity as r approaches infinity. Use radial force balance on a small parcel moving in a circular path, and define the central pressure deficit as Delta P=P_infinity-P(0).",
    ),
    "A3:vortex_b": staticization_spec(
        "Isothermal Rankine setup; displacement argument analogous to Updraft A(c); circulation gamma(r)=r v(r); stability and application requested.",
        "For Part B, treat the atmosphere as isothermal with uniform density rho0. Use a radial displacement argument analogous to the parcel-displacement argument requested in Updraft A(c). Define the circulation variable gamma(r)=r v(r), with v=Omega r for r<R and v=Omega R^2/r for r>R.",
    ),
    "B1:a": staticization_spec(
        "Uniform hydrogen cloud, rho_H, central star rate Q_*, steady ionization-recombination balance, R=n_p n_e alpha, n=rho/m, full ionization; r_s requested.",
        "A star at the center of a uniform hydrogen cloud of mass density rho_H emits ionizing photons at constant rate Q_* (photons/s). The ionized region is in steady state: total ionization rate equals total recombination rate. The recombination rate per unit volume is R=n_p n_e alpha, and number density is defined by n=rho/m. Assume the sphere is completely ionized and contains only hydrogen.",
    ),
    "B1:b_i": staticization_spec(
        "Early ionization front; gas initially at rest; thin discontinuity; recombinations behind front neglected; s and J definitions; speed relation requested.",
        "When the star first turns on, an ionization front sweeps outward. In this early stage the gas is initially at rest, the front is a thin discontinuity, and recombinations behind it are neglected. Let s be the front speed into neutral hydrogen of number density n_H, and let J be the ionizing-photon flux reaching the front in photons per unit area per unit time.",
    ),
    "B1:b_ii": staticization_spec(
        "Early-front assumptions; star photon rate Q_*; s and J definitions from part (b); spherical radius r(t) with r(0)=0; no recombinations.",
        "When the star first turns on, a spherical ionization front sweeps outward through gas initially at rest. Treat the front as a thin discontinuity and neglect recombinations behind it. The star emits Q_* photons per second. In part (b), s denotes the front speed into neutral hydrogen of density n_H and J denotes the photon flux reaching the front.",
    ),
    "B1:c_i": staticization_spec(
        "Tin about 10^4 K, Tout about 100 K, initially similar hydrogen number densities, ideal gas, fully ionized interior, maintained ionization balance; pressure ratio requested.",
        "Inside the sphere Tin is approximately 10^4 K, while the neutral exterior has Tout approximately 100 K. Initially n_H,in is approximately n_H,out. Treat the gas as ideal; the expanding interior remains fully ionized, and ionization balance is maintained.",
    ),
    "B1:c_ii": staticization_spec(
        "Complete bubble temperature/density/ideal-gas/full-ionization/ionization-balance setup; r_s meaning; final pressure-equilibrium radius requested.",
        "Inside the sphere Tin is approximately 10^4 K, while the neutral exterior has Tout approximately 100 K. Initially n_H,in is approximately n_H,out. Treat the gas as ideal; the expanding interior remains fully ionized, and ionization balance is maintained. The symbol r_s is the Stroemgren radius requested in part (a).",
    ),
    "B1:d_i": staticization_spec(
        "Cluster hot gas T about 10^7 K; hydrostatic equation dP/dr=-rho dPhi/dr; Phi per unit mass; ideal approximately isothermal gas; dPhi/dr=v_c^2/r; rho power law requested.",
        "Galaxy clusters contain hot gas with T approximately 10^7 K in hydrostatic equilibrium, dP/dr=-rho dPhi/dr, where Phi is gravitational potential per unit mass. Treat the gas as ideal and approximately isothermal. For part (d), dPhi/dr=v_c^2/r with constant v_c.",
    ),
    "B1:d_ii_i": staticization_spec(
        "Cluster hydrostatic/ideal/isothermal setup; n=rho/m definition; free-free volumetric rate Lambda proportional to n_e^2 sqrt(T); t_cool exponent request.",
        "For the hot, ideal, approximately isothermal cluster gas, number density is related to particle mass density by n=rho/m. The gas cools by free-free X-ray emission at volumetric rate Lambda proportional to n_e^2 sqrt(T).",
    ),
    "B1:d_ii_ii": staticization_spec(
        "Cluster ideal/isothermal setup; prior requested rho(r) proportional to r^-beta; free-free rate; preceding t_cool proportional to T^a rho_e^b ansatz; radial behavior request.",
        "For the hot, ideal, approximately isothermal cluster gas, part (d)(i) asks one to show rho(r) is proportional to r^-beta. Free-free X-ray emissivity per volume obeys Lambda proportional to n_e^2 sqrt(T). Part (d)(ii)(i) defines the cooling-time scaling ansatz t_cool proportional to T^a rho_e^b.",
    ),
    "B1:d_iii": staticization_spec(
        "Cluster hydrostatic/ideal/isothermal density model; rho(r) power-law request; free-free cooling model and radial cooling-time request; observed hot center; expectation/explanation requested.",
        "The cluster model assumes hot ideal approximately isothermal gas in hydrostatic equilibrium, dP/dr=-rho dPhi/dr, with dPhi/dr=v_c^2/r. Part (d)(i) asks for a power-law density rho(r) proportional to r^-beta. Part (d)(ii) gives free-free emissivity Lambda proportional to n_e^2 sqrt(T) and asks for the cooling time and its radial limits. Observations show luminous X-rays from hot gas at cluster centers.",
    ),
    "B2:a": staticization_spec(
        "Fixed +1 C center; attractive conic equation; three equal-time positions and -1 C charge; maximum radius requested."
    ),
    "B2:b": staticization_spec(
        "Fixed +1 C center; repulsive conic equation; three equal-time positions and +2 C charge; large-time velocity direction from +x requested."
    ),
    "B2:c_i": staticization_spec(
        "Identical +2 C particles; common v_infinity; incoming lines parallel to part-(b) initial asymptote; one-at-a-time/no mutual interaction; B and physical velocity-deflection definitions; conic derivation requested.",
        "For part (c), identical +2 C particles approach from infinity with common speed v_infinity along lines parallel to the initial asymptotic direction of the particle in part (b). They are injected one at a time and do not interact with one another. B is the perpendicular distance from the incoming straight-line path to the origin. The deflection angle alpha_s is the angle between incoming and outgoing asymptotic velocity directions.",
    ),
    "B2:c_ii": staticization_spec(
        "Complete scattering-family/B/deflection definitions; reference trajectory's +2 C charge and three equal-time coordinates from part (b); expression in terms of its r0 requested.",
        "For part (c), identical +2 C particles approach from infinity with common speed v_infinity along parallel incoming asymptotes; they are injected one at a time and do not interact. B is the incoming-line distance to the origin, and alpha_s is the angle between incoming and outgoing velocity directions. The reference trajectory from part (b) is the +2 C particle whose three equal-time positions are (3 m,-4 m), (2 m,0), and (3 m,4 m); part (b) asks for its large-time velocity direction. Use only this question-side information and prior-turn work.",
    ),
    "B3:a": staticization_spec(
        "Energy definition of Q; inductor L with resistance R; I(t)=I0 cos(omega t); expression for Q requested."
    ),
    "B3:b": staticization_spec(
        "Energy definition of Q; long superconducting solenoid L; tightly wound shell a,t<<a,sigma,ell; no end/skin effects; current; requested variables."
    ),
    "B3:c": staticization_spec(
        "Complete conducting-cylinder setup; cross-reference to part (a)'s ordinary lossy-inductor question; effective resistance requested without supplying the prior answer.",
        "Part (a) considers an inductor of inductance L and resistance R carrying I(t)=I0 cos(omega t), and asks for Q from the energy definition. In the present conducting-cylinder setup, use that question-side comparison to determine an effective resistance. (The source labels this cross-reference as 'part (1)'; context identifies part (a).)",
    ),
    "B3:d": staticization_spec(
        "Complete conducting-cylinder setup; external series resistor R; total quality factor requested."
    ),
    "B3:e": staticization_spec(
        "Frequency comparison between ideal L plus series R and same circuit with cylinder; physical explanation requested.",
        "Compare (1) an RL circuit made of an ideal inductor L in series with a resistor R and (2) the same circuit with the conducting cylinder inserted. Explain physically why their frequency dependence differs.",
    ),
}


ITEMS = [
    # A1: benchmark-derived 25-point normalization.
    dict(problem="A1", part="a", points=5.0, qpages=[1, 2], spages=[2, 3],
         prompt="Find both the adjustable focal-length range f and optical-power range P of the internal lens.",
         answer="P_eye is 10.1 to 11.0 D, corresponding to f=9.90 to 9.09 cm (reported as the interval [9.09,9.90] cm).",
         solution="Use 1/f=1/s+1/s' with s'=0.10 m. The endpoints s=10.0 m and 1.00 m give 10.1 D and 11.0 D; invert them for the focal-length endpoints.",
         criteria=[criterion("a1_a_lens_equation",1.5,"Use the thin-lens relation with the fixed 0.10 m image distance.",2,3),criterion("a1_a_power_endpoints",2.0,"Evaluate the two object-distance endpoints as 10.1 D and 11.0 D and order the range correctly.",2,3),criterion("a1_a_focal_range",1.5,"Invert the power endpoints to obtain f in [9.09,9.90] cm.",3)]),
    dict(problem="A1", part="b", points=4.0, qpages=[1, 2], spages=[3],
         prompt="Find the range of corrective-lens optical powers that allows the robot to observe stars at infinity.",
         answer="P_corr is in [-1.0,-0.10] D.",
         solution="The corrective lens must form a virtual image at L=1.00 to 10.0 m. With s=infinity, P_corr=-1/L.",
         criteria=[criterion("a1_b_virtual_image",2.0,"Recognize that the corrective lens must form a virtual image inside the eye's 1-10 m focusing range.",3),criterion("a1_b_power_range",2.0,"Apply P=-1/L and give [-1.0,-0.10] D with the correct sign.",3)]),
    dict(problem="A1", part="c", points=4.0, qpages=[1, 2], spages=[3, 4],
         prompt="Can one corrective lens allow both star observation and reading at 25.0 cm? Explain.",
         answer="No. Reading needs P_corr in [3.0,3.9] D, while star observation needs [-1.0,-0.10] D; the intervals do not overlap.",
         solution="Reading at 25 cm requires total power 1/0.25+1/0.10=14 D. Subtracting the 10.1-11.0 D internal range gives 3.0-3.9 D, disjoint from the negative star-correction range.",
         criteria=[criterion("a1_c_reading_total",1.0,"Find the 14 D total power required for a 25 cm object and 10 cm image.",4),criterion("a1_c_reading_correction",1.5,"Obtain the corrective range [3.0,3.9] D.",4),criterion("a1_c_disjoint",1.5,"Compare with the star range and conclude correctly that no single lens works.",4)]),
    dict(problem="A1", part="d", points=5.0, qpages=[1, 2], spages=[4, 5],
         prompt="Estimate the diffraction-limited maximum print resolution in dpi at 25.0 cm for wavelength 550 nm.",
         answer="About 760 dpi.",
         solution="Use theta about 1.22 lambda/d=1.34e-4 rad, delta x=L theta=3.35e-5 m, and dpi=0.0254/delta x about 760.",
         criteria=[criterion("a1_d_rayleigh",2.0,"Use a diffraction angle of order 1.22 lambda/d.",4),criterion("a1_d_spacing",1.5,"Convert the angle to a page spacing near 3.35e-5 m at 0.25 m.",4),criterion("a1_d_dpi",1.5,"Convert metres per dot to approximately 760 dots per inch.",5)]),
    dict(problem="A1", part="e_i", points=4.0, qpages=[1, 2, 3], spages=[5],
         prompt="For stellar imaging with collected optical power P0, estimate the sensor intensity in terms of P0, wavelength lambda, and eye parameters.",
         answer="Up to an order-unity Airy factor, I is of order P0 d^2/(lambda^2 s'^2).",
         solution="The diffraction spot radius scales as lambda s'/d, its area as lambda^2 s'^2/d^2, and intensity is collected power divided by this area.",
         criteria=[criterion("a1_ei_spot",1.5,"Relate spot radius to lambda s'/d.",5),criterion("a1_ei_area",1.0,"Square the spot scale to obtain its area dependence.",5),criterion("a1_ei_intensity",1.5,"Obtain I proportional to P0 d^2/(lambda^2 s'^2), allowing order-unity Airy constants.",5)]),
    dict(problem="A1", part="e_ii", points=3.0, qpages=[1, 2, 3], spages=[5],
         prompt="For the same stellar object, by what factor does intensity change when pupil diameter falls from 5.0 mm to 2.0 mm?",
         answer="I_new/I_old=(2/5)^4=16/625 about 0.0256, so intensity decreases by about a factor of 39.",
         solution="Collected power scales as d^2 and diffraction spot area as d^-2, hence intensity scales as d^4.",
         criteria=[criterion("a1_eii_scaling",1.5,"Combine collection and diffraction to get I proportional to d^4.",5),criterion("a1_eii_factor",1.5,"Evaluate (2/5)^4=16/625, or an equivalent decrease factor about 39.",5)]),

    # A2.
    dict(problem="A2", part="a", points=7.0, qpages=[1, 4], spages=[6],
         prompt="Find the normal force at t=0.",
         answer="The magnitude is N=(m v^2/r0) cot(alpha), directed normal to the cone.",
         solution="Resolve the constrained acceleration normal to the cone (or use the normal curvature of the cone) to obtain N/m=(v^2/r0)cot(alpha). The provided solution's intermediate N/cos(alpha) component wording is inconsistent, but its final formula is correct.",
         criteria=[criterion("a2_a_geometry",2.0,"Identify the circle radius r0 sin(alpha) and the cone-normal direction or an equivalent differential-geometric setup.",6),criterion("a2_a_constraint_accel",3.0,"Correctly remove the component tangent to the cone / compute normal curvature.",6),criterion("a2_a_force",2.0,"Give N=(m v^2/r0)cot(alpha) with a physically correct direction.",6)], defects=["usapho2026_a2a_component_wording"]),
    dict(problem="A2", part="b", points=6.0, qpages=[1, 4], spages=[6, 7],
         prompt="At B, find the angle psi between the velocity and OB.",
         answer="psi=arcsin(r0/r1) on the outgoing branch.",
         solution="Energy keeps the speed v constant and axial angular momentum gives m(r0 sin alpha)v=m(r1 sin alpha)v sin psi.",
         criteria=[criterion("a2_b_conservation",2.5,"Use conservation of axial angular momentum and constant speed.",6,7),criterion("a2_b_components",1.5,"Relate the azimuthal velocity at B to v sin(psi) and radius to r1 sin(alpha).",6,7),criterion("a2_b_angle",2.0,"Obtain sin(psi)=r0/r1 and select the outgoing acute branch.",7)]),
    dict(problem="A2", part="c", points=5.0, qpages=[1, 4], spages=[7],
         prompt="At what time t does the mass reach B?",
         answer="t=sqrt(r1^2-r0^2)/v.",
         solution="Unroll the cone. The geodesic becomes a straight tangent line at speed v, forming a right triangle with radii r0 and r1.",
         criteria=[criterion("a2_c_unroll",2.0,"Unroll the cone and recognize straight-line constant-speed motion.",7),criterion("a2_c_distance",1.5,"Use the right triangle to get path length sqrt(r1^2-r0^2).",7),criterion("a2_c_time",1.5,"Divide by v to obtain the stated time.",7)]),
    dict(problem="A2", part="d", points=7.0, qpages=[1, 4], spages=[7, 8],
         prompt="After a long time, find the total number of revolutions about the cone axis.",
         answer="N_rev=1/(4 sin(alpha)).",
         solution="In the unrolled plane the tangent trajectory sweeps beta=pi/2 as t tends to infinity. Since beta=phi sin alpha, Delta phi=pi/(2 sin alpha) and divide by 2pi.",
         criteria=[criterion("a2_d_tangent_angle",2.0,"Recognize the asymptotic pi/2 swept angle in the unrolled plane.",7),criterion("a2_d_angle_map",2.5,"Use beta=phi sin(alpha) for the cone development.",7),criterion("a2_d_revolutions",2.5,"Convert Delta phi to 1/(4 sin alpha) revolutions.",8)]),

    # A3.
    dict(problem="A3", part="updraft_a", points=3.0, qpages=[1, 5], spages=[9],
         prompt="Updraft A(a): derive dP/dz in terms of atmospheric density rho.",
         answer="dP/dz=-rho g.", solution="Vertical force balance on a thin air slab gives hydrostatic equilibrium.",
         criteria=[criterion("a3_ua_slab",1.5,"Set up vertical force balance on a thin atmospheric slab.",9),criterion("a3_ua_gradient",1.5,"Obtain dP/dz=-rho g with the correct sign.",9)]),
    dict(problem="A3", part="updraft_b", points=5.0, qpages=[1, 5], spages=[9],
         prompt="Updraft A(b): derive the parcel's adiabatic lapse rate Gamma_p=-dT_p/dz.",
         answer="Gamma_p=mu g/C_P.", solution="Use dU=-P dV, U'=nC_V, differentiate PV=nRT, and combine nC_P dT=V dP with hydrostatic balance and m=n mu.",
         criteria=[criterion("a3_ub_first_law",1.5,"Apply the adiabatic first law dU=-P dV.",9),criterion("a3_ub_cp_relation",1.5,"Combine with the ideal-gas differential to obtain n C_P dT=V dP.",9),criterion("a3_ub_hydrostatic",1.0,"Insert dP=-rho g dz and m=n mu.",9),criterion("a3_ub_lapse",1.0,"Give Gamma_p=mu g/C_P.",9)]),
    dict(problem="A3", part="updraft_c_i", points=3.0, qpages=[1, 5], spages=[10],
         prompt="Updraft A(c)(i): determine the condition on environmental lapse rate Gamma_env for instability.",
         answer="The atmosphere is unstable when Gamma_env>Gamma_p.", solution="After upward displacement, compare parcel and environmental temperatures/densities at equal pressure; buoyancy reinforces displacement if the environment cools faster.",
         criteria=[criterion("a3_uci_density_compare",1.5,"Compare parcel and environment density at common pressure after a small displacement.",10),criterion("a3_uci_condition",1.5,"Conclude instability for Gamma_env>Gamma_p.",10)]),
    dict(problem="A3", part="updraft_c_ii", points=4.0, qpages=[1, 5], spages=[10],
         prompt="Updraft A(c)(ii): find the oscillation frequency when the atmosphere is stable.",
         answer="The angular frequency is omega_B=sqrt[(g/T0)(Gamma_p-Gamma_env)].",
         solution="Linearize buoyancy: z_ddot=(g/T0)(Gamma_env-Gamma_p) z, which is simple harmonic motion in the stable regime.",
         criteria=[criterion("a3_ucii_buoyancy",1.5,"Derive the linear acceleration proportional to (Gamma_env-Gamma_p) delta z.",10),criterion("a3_ucii_shm",1.0,"Identify the stable-sign equation as simple harmonic motion.",10),criterion("a3_ucii_frequency",1.5,"Give sqrt[(g/T0)(Gamma_p-Gamma_env)], noting it is angular frequency.",10)]),
    dict(problem="A3", part="vortex_a", points=5.0, qpages=[1, 5], spages=[10, 11],
         prompt="Vortex B(a): derive the radial pressure gradient and the Rankine-vortex central pressure deficit.",
         answer="dP/dr=rho0 v^2/r and Delta P=P_infinity-P(0)=rho0 Omega^2 R^2.",
         solution="Radial balance gives an outward-increasing pressure. Integrating the outer 1/r velocity and inner solid-body velocity, matched at R, gives two equal half-deficits. The source's first finite-difference balance has a sign inconsistency, but its differential and final results are correct.",
         criteria=[criterion("a3_va_gradient",2.0,"Obtain dP/dr=rho0 v^2/r with pressure increasing outward.",10),criterion("a3_va_outer",1.0,"Integrate the outer velocity profile from infinity to R.",10),criterion("a3_va_inner",1.0,"Integrate the solid-body core and match pressure at R.",10,11),criterion("a3_va_deficit",1.0,"Give Delta P=rho0 Omega^2 R^2.",11)], defects=["usapho2026_a3_vortex_force_balance_sign"]),
    dict(problem="A3", part="vortex_b", points=5.0, qpages=[1, 5], spages=[11],
         prompt="Vortex B(b): derive a circulation stability condition and apply it to the Rankine vortex.",
         answer="Rayleigh stability requires d(gamma^2)/dr>0; for the given positive circulation this is d gamma/dr>0. The solid-body core is stable and the outer irrotational region is neutral.",
         solution="A radially displaced parcel conserves gamma=rv. Compare its required centripetal acceleration with that supplied by the environmental pressure gradient. In the given positive-Omega case increasing gamma is restoring.",
         criteria=[criterion("a3_vb_conserve",1.5,"Use conservation of parcel angular momentum/circulation under radial displacement.",11),criterion("a3_vb_compare",1.5,"Compare parcel and environmental azimuthal speeds or centripetal requirements.",11),criterion("a3_vb_rayleigh",1.0,"State d(gamma^2)/dr>0, or d gamma/dr>0 with the positive-circulation qualification.",11),criterion("a3_vb_apply",1.0,"Classify the core stable and exterior neutral.",11)]),

    # B1.
    dict(problem="B1", part="a", points=4.0, qpages=[6, 7], spages=[12],
         prompt="Derive the Stromgren radius r_s in terms of Q_*, alpha, and n_H.", answer="r_s=[3Q_*/(4 pi alpha n_H^2)]^(1/3).",
         solution="For fully ionized hydrogen n_p=n_e=n_H. Set Q_* equal to alpha n_H^2 times the sphere volume.",
         criteria=[criterion("b1_a_density",1.0,"Set n_p=n_e=n_H for fully ionized hydrogen.",12),criterion("b1_a_balance",1.5,"Set total recombinations equal to Q_*.",12),criterion("b1_a_radius",1.5,"Solve for the stated cube-root radius.",12)]),
    dict(problem="B1", part="b_i", points=2.0, qpages=[6, 7], spages=[12],
         prompt="Early front B(b)(i): derive its speed s in terms of photon flux J and n_H.", answer="s=J/n_H.", solution="Equate atoms swept per area per time, n_H s, to ionizing photons per area per time J.",
         criteria=[criterion("b1_bi_balance",1.0,"Equate swept neutral atoms and incident photons.",12),criterion("b1_bi_speed",1.0,"Obtain s=J/n_H.",12)]),
    dict(problem="B1", part="b_ii", points=3.0, qpages=[6, 7], spages=[12, 13],
         prompt="Early front B(b)(ii): find r(t) for r(0)=0, neglecting recombinations.", answer="r(t)=[3Q_* t/(4 pi n_H)]^(1/3).",
         solution="Use J=Q_*/(4 pi r^2) and dr/dt=J/n_H, then integrate r^2 dr.",
         criteria=[criterion("b1_bii_flux",1.0,"Use inverse-square photon flux at the front.",12,13),criterion("b1_bii_ode",1.0,"Set dr/dt=Q_*/(4 pi n_H r^2).",13),criterion("b1_bii_integrate",1.0,"Integrate with r(0)=0 to get the cube-root law.",13)]),
    dict(problem="B1", part="c_i", points=2.0, qpages=[6, 7], spages=[13],
         prompt="Bubble B(c)(i): calculate P_in/P_out for the stated initial temperatures and similar hydrogen number densities.", answer="P_in/P_out=2T_in/T_out about 200.", solution="Fully ionized hydrogen has a proton and electron per atom, whereas neutral hydrogen has one gas particle per atom.",
         criteria=[criterion("b1_ci_particle_count",1.0,"Include the factor of two in ionized-gas particle density.",13),criterion("b1_ci_ratio",1.0,"Give 2T_in/T_out and evaluate about 200.",13)]),
    dict(problem="B1", part="c_ii", points=3.0, qpages=[6, 7], spages=[13],
         prompt="Bubble B(c)(ii): find the pressure-equilibrium final radius in terms of r_s and T_in/T_out.", answer="r_final=r_s(2T_in/T_out)^(2/3).",
         solution="Pressure balance gives n_in/n_out=T_out/(2T_in). Ionization balance gives r proportional to n^-2/3.",
         criteria=[criterion("b1_cii_pressure",1.0,"Use pressure balance to obtain n_in/n_out=T_out/(2T_in).",13),criterion("b1_cii_scaling",1.0,"Use r proportional to n^-2/3 from ionization balance.",13),criterion("b1_cii_radius",1.0,"Combine them into r_s(2T_in/T_out)^(2/3).",13)]),
    dict(problem="B1", part="d_i", points=4.0, qpages=[6, 7, 8], spages=[13, 14],
         prompt="Cluster B(d)(i): show rho(r) is proportional to r^-beta and find beta.", answer="beta=m_p v_c^2/(2 k_B T) for fully ionized hydrogen under the stated approximation.",
         solution="Use P=2n_H k_B T and rho approximately m_p n_H in hydrostatic balance with dPhi/dr=v_c^2/r, then integrate d ln n_H/d ln r.",
         criteria=[criterion("b1_di_eos",1.0,"Use the fully ionized-hydrogen pressure and mass density consistently.",13,14),criterion("b1_di_ode",1.5,"Derive d ln n_H/dr=-m_p v_c^2/(2k_BT r).",14),criterion("b1_di_beta",1.5,"Integrate to the power law with the stated beta.",14)]),
    dict(problem="B1", part="d_ii_i", points=2.0, qpages=[6, 7, 8], spages=[14],
         prompt="Cluster B(d)(ii)(i): if t_cool is proportional to T^a rho_e^b, find a and b.", answer="a=1/2 and b=-1.", solution="Thermal energy density is proportional to rho_e T while free-free power is proportional to rho_e^2 sqrt(T), so their ratio is sqrt(T)/rho_e.",
         criteria=[criterion("b1_diii_ratio",1.0,"Form cooling time as thermal energy density divided by free-free loss rate.",14),criterion("b1_diii_exponents",1.0,"Obtain a=1/2 and b=-1.",14)]),
    dict(problem="B1", part="d_ii_ii", points=2.0, qpages=[6, 7, 8], spages=[14],
         prompt="Cluster B(d)(ii)(ii): express t_cool(r) and give its limits as r approaches 0 and infinity.", answer="For isothermal gas, t_cool(r) is proportional to sqrt(T) r^beta; it tends to 0 at the center and infinity at large radius.",
         solution="Substitute rho_e proportional to r^-beta into t_cool proportional to sqrt(T)/rho_e.",
         criteria=[criterion("b1_diiii_radial",1.0,"Obtain t_cool proportional to sqrt(T) r^beta.",14),criterion("b1_diiii_limits",1.0,"State the correct central and outer limits for beta>0.",14)], defects=["usapho2026_b1_duplicate_as"]),
    dict(problem="B1", part="d_iii", points=3.0, qpages=[6, 7, 8], spages=[14],
         prompt="Cluster B(d)(iii): is observed hot central gas expected in this model? Explain.", answer="No. The model predicts vanishingly short central cooling time, so persistent hot gas requires omitted heating/feedback physics.",
         solution="The density cusp makes t_cool tend to zero, contradicting long-lived observed hot gas; an additional heating mechanism such as feedback is needed.",
         criteria=[criterion("b1_diii_model",1.5,"Connect the central density/cooling-time behavior to rapid predicted cooling.",14),criterion("b1_diii_conclusion",1.5,"Conclude the observation is not expected and identify missing heating/feedback qualitatively.",14)]),

    # B2. Parts c(i)-c(ii) use a curator-corrected result; see source defects.
    dict(problem="B2", part="a", points=7.0, qpages=[6, 9], spages=[15, 16],
         prompt="For the -1 C particle at (0,-5 m), (3 m,0), (0,5 m) at equal intervals, find its maximum distance from the origin.", answer="15 m.",
         solution="Symmetry puts pericenter at (3,0). At phi=plus/minus pi/2, r=r0=5 m. Then 3=5/(1+e), so e=2/3 and apocenter is 5/(1-e)=15 m.",
         criteria=[criterion("b2_a_symmetry",1.5,"Use equal-time symmetry to identify (3,0) as pericenter.",15),criterion("b2_a_r0",1.5,"Use the y-axis points to obtain r0=5 m.",15),criterion("b2_a_e",1.5,"Obtain eccentricity e=2/3.",15,16),criterion("b2_a_apocenter",2.5,"Evaluate rmax=r0/(1-e)=15 m.",16)]),
    dict(problem="B2", part="b", points=8.0, qpages=[6, 9], spages=[16, 17],
         prompt="For the +2 C particle at (3,-4 m), (2 m,0), (3,4 m) at equal intervals, find the large-time velocity angle theta from +x.", answer="theta=arccos(1/3) about 70.5 degrees.",
         solution="Symmetry gives periapsis 2 m. Using the 5 m point with cos phi=3/5 in the repulsive conic yields e=3 and r0=4 m. The outgoing asymptote has cos phi=1/e.",
         criteria=[criterion("b2_b_symmetry",1.5,"Identify (2,0) as closest approach.",16),criterion("b2_b_periapsis",1.5,"Use r0/(e-1)=2.",16),criterion("b2_b_other_point",2.0,"Use r=5 and cos phi=3/5 to solve e=3 and r0=4 m.",16,17),criterion("b2_b_asymptote",1.5,"Set e cos phi-1=0 at infinity.",17),criterion("b2_b_angle",1.5,"Give arccos(1/3) about 70.5 degrees for the outgoing velocity.",17)]),
    dict(problem="B2", part="c_i", points=5.0, qpages=[6, 9], spages=[17, 18],
         prompt="Scattering B(c)(i): derive B(alpha_s) for the family, where alpha_s is the angle between incoming and outgoing velocity directions.",
         answer="B(alpha_s)=a_C cot(alpha_s/2), where a_C=k q_alpha q_gamma/(m_gamma v_inf^2).",
         source_answer="B(alpha)=r0 cot(alpha/2).",
         source_answer_scope="The provided solution presents this single final formula for the combined parts B2(c)(i)-(ii).",
         source_solution_summary="The provided solution sets alpha=2 arccos(1/e), uses B=r0/sqrt(e^2-1), and concludes B(alpha)=r0 cot(alpha/2).",
         adjudication_id="usapho2026_b2c_scattering_20260828",
         solution="For repulsive Coulomb scattering, e=sqrt(1+(B/a_C)^2) and the physical velocity deflection is alpha_s=2 arcsin(1/e), giving the Rutherford relation. The provided solution instead uses the supplementary angle 2 arccos(1/e), contrary to the question's velocity-direction definition.",
         criteria=[criterion("b2_ci_energy_scale",1.5,"Introduce the fixed-energy Coulomb length a_C=k q_alpha q_gamma/(m_gamma v_inf^2), or an equivalent conserved-energy/angular-momentum relation.",17,18),criterion("b2_ci_eccentricity",1.5,"Relate e^2-1 to (B/a_C)^2.",17,18),criterion("b2_ci_deflection",1.0,"Use the physical deflection alpha_s=2 arcsin(1/e), not its supplement.",18),criterion("b2_ci_rutherford",1.0,"Obtain B=a_C cot(alpha_s/2).",18)], defects=["usapho2026_b2c_scattering_solution", "usapho2026_b2c_cross_reference"]),
    dict(problem="B2", part="c_ii", points=5.0, qpages=[6, 9], spages=[17, 18],
         prompt="Scattering B(c)(ii): using the reference trajectory from part (b), express B(alpha_s) in terms of that trajectory's r0.",
         answer="Since e=3 for the reference trajectory, a_C=r0/(e^2-1)=r0/8, so B(alpha_s)=(r0/8) cot(alpha_s/2).",
         source_answer="B(alpha)=r0 cot(alpha/2).",
         source_answer_scope="The provided solution presents this single final formula for the combined parts B2(c)(i)-(ii).",
         source_solution_summary="The provided solution sets alpha=2 arccos(1/e), uses B=r0/sqrt(e^2-1), and concludes B(alpha)=r0 cot(alpha/2).",
         adjudication_id="usapho2026_b2c_scattering_20260828",
         solution="For a repulsive conic the semilatus rectum is r0=a_C(e^2-1). Part (b) gives e=3 and r0=4 m, fixing the common energy scale a_C=r0/8. The provided solution omits this factor and treats the orbit-dependent r0 as common to the family.",
         criteria=[criterion("b2_cii_semilatus",2.0,"Relate the reference semilatus rectum to the common Coulomb length: r0=a_C(e^2-1).",17,18),criterion("b2_cii_factor",1.5,"Use e=3 to obtain a_C=r0/8.",17,18),criterion("b2_cii_result",1.5,"Give B=(r0/8)cot(alpha_s/2) under the prompt's physical deflection convention.",18)], defects=["usapho2026_b2c_scattering_solution", "usapho2026_b2c_cross_reference"]),

    # B3.
    dict(problem="B3", part="a", points=4.0, qpages=[6, 10], spages=[19],
         prompt="For an inductor L with resistance R carrying I0 cos(omega t), derive Q from the energy definition.", answer="Q=omega L/R.",
         solution="Umax=LI0^2/2, average resistive power=RI0^2/2, and loss per period is pi R I0^2/omega.",
         criteria=[criterion("b3_a_energy",1.0,"Find Umax=LI0^2/2.",19),criterion("b3_a_loss",1.5,"Find average power and energy lost per cycle.",19),criterion("b3_a_q",1.5,"Obtain Q=omega L/R.",19)]),
    dict(problem="B3", part="b", points=9.0, qpages=[6, 10], spages=[19, 20],
         prompt="For the superconducting solenoid around the conducting shell, determine Q in terms of omega, sigma, t, a, and L.", answer="Q=2/(mu0 sigma t a omega); L cancels after using the long-solenoid inductance relation.",
         solution="Use B=mu0(N/ell)I and Faraday emf through pi a^2. The shell resistance is 2pi a/(sigma t ell). Compute eddy-current power, loss per cycle, and divide LI0^2/2 by it; substituting L=mu0 N^2 pi a^2/ell yields the result.",
         criteria=[criterion("b3_b_flux",2.0,"Obtain the solenoid field, flux, and induced-emf amplitude.",19,20),criterion("b3_b_shell_resistance",1.5,"Use shell path resistance 2pi a/(sigma t ell).",20),criterion("b3_b_current_power",2.0,"Compute induced-current amplitude and average eddy-current loss.",20),criterion("b3_b_cycle_loss",1.5,"Convert average loss to energy per cycle and use Umax=LI0^2/2.",20),criterion("b3_b_q",2.0,"Use the solenoid inductance relation to obtain Q=2/(mu0 sigma t a omega).",20)]),
    dict(problem="B3", part="c", points=3.0, qpages=[6, 10], spages=[20, 21],
         prompt="Determine the effective resistance associated with the conducting cylinder.", answer="R_eff=(1/2) mu0 sigma t a L omega^2.",
         solution="Match Q=omega L/R_eff to the cylinder-only quality factor.",
         criteria=[criterion("b3_c_match",1.0,"Use the equivalent RL relation Q=omega L/R_eff.",20,21),criterion("b3_c_result",2.0,"Obtain R_eff=(1/2)mu0 sigma t a L omega^2.",21)], defects=["usapho2026_b3c_cross_reference"]),
    dict(problem="B3", part="d", points=3.0, qpages=[6, 10], spages=[21],
         prompt="With an external series resistor R, determine the total Q in the presence of the cylinder.", answer="Q=omega L/[R+(1/2)mu0 sigma t a L omega^2].",
         solution="The external and eddy-current effective resistances add in the loss term.",
         criteria=[criterion("b3_d_add",1.0,"Add R and R_eff as loss-equivalent series resistances.",21),criterion("b3_d_q",2.0,"Give the complete Q expression.",21)]),
    dict(problem="B3", part="e", points=6.0, qpages=[6, 10], spages=[21, 22],
         prompt="Compare Q's frequency dependence with and without the cylinder and explain the physics.",
         answer="Without the cylinder Q=omega L/R grows linearly. With it, Q=omega L/[R+C omega^2], C=(1/2)mu0 sigma t a L: Q grows as omega at low frequency, peaks, then falls as 1/omega because eddy-current dissipation grows as omega^2.",
         solution="The induced emf and eddy-current amplitude are proportional to omega, so dissipated power and the effective resistance scale as omega^2.",
         criteria=[criterion("b3_e_plain",1.0,"State the ordinary RL result and linear growth.",21),criterion("b3_e_full",1.5,"Use Q=omega L/(R+C omega^2) for the cylinder case.",21),criterion("b3_e_low",1.0,"Identify low-frequency Q proportional to omega.",21),criterion("b3_e_high",1.0,"Identify high-frequency Q proportional to 1/omega and an intervening maximum.",21,22),criterion("b3_e_physics",1.5,"Explain that Faraday-induced eddy-current loss grows as omega^2.",22)]),
]


SOURCE_DEFECTS = [
    {
        "id": "usapho2026_no_official_points_or_rubric",
        "severity": "evaluation_limitation",
        "location": "both provided PDFs, complete inspection",
        "description": "No point allocations, grading rubric, score, or credit instructions are printed. The benchmark uses an explicit curator normalization of 25 points per full problem and derived criterion weights.",
    },
    {
        "id": "usapho2026_a2a_component_wording",
        "severity": "source_derivation_inconsistency",
        "location": "solutions PDF page 6",
        "description": "The intermediate component statement writes N/cos(alpha)=m v^2/(r0 sin(alpha)) while describing a horizontal component; that wording/equation is inconsistent. The final N=(m v^2/r0)cot(alpha) is correct from the cone-normal acceleration.",
    },
    {
        "id": "usapho2026_a3_vortex_force_balance_sign",
        "severity": "source_derivation_inconsistency",
        "location": "solutions PDF page 10",
        "description": "The finite-difference radial balance as printed has a sign inconsistent with its following correct result dP/dr=rho0 v^2/r. The final pressure deficit is correct.",
    },
    {
        "id": "usapho2026_b1_duplicate_as",
        "severity": "editorial",
        "location": "problems PDF page 8 (printed page 7) and solutions PDF page 14",
        "description": "B1(d)(ii)(ii) asks for behavior 'as as r -> 0'; the duplicated word has no mathematical effect.",
    },
    {
        "id": "usapho2026_b2c_cross_reference",
        "severity": "editorial_ambiguity",
        "location": "problems PDF page 9 (printed page 8) and solutions PDF page 17",
        "description": "B2(c)(ii) says 'part (2)' although the problem is lettered; context indicates the reference trajectory from part (b).",
    },
    {
        "id": "usapho2026_b2c_scattering_solution",
        "severity": "major_scientific_error",
        "location": "solutions PDF pages 17-18",
        "description": "The solution equates the physical velocity deflection with 2 arccos(1/e), its supplement, and treats the orbit-dependent semilatus rectum r0 as common across a fixed-energy impact-parameter family. Correctly, B=a_C cot(alpha_s/2), and part (b)'s e=3 gives a_C=r0/8, hence B=(r0/8)cot(alpha_s/2).",
    },
    {
        "id": "usapho2026_b3c_cross_reference",
        "severity": "editorial",
        "location": "problems PDF page 10 (printed page 9) and solutions PDF page 20",
        "description": "B3(c) says 'part (1)' although the problem is lettered; it refers to the Q=omega L/R result in part (a).",
    },
]

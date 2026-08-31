"""Human-audited, solver-safe visual transcriptions for IChO 2026.

Every string in this module was written from the official G0, question-booklet,
and answer-booklet page renders only.  No official solution/marking PDF was
consulted while composing this solver-side material.  The descriptions encode
non-text layout, chemical connectivity, stereochemical marks, plots, tables,
and response templates that the PDF text layer omits.
"""

COMMON = {
    "plain": (
        "Human visual check: the cited pages contain no task-relevant non-text "
        "information beyond typographic layout, blank working space, and the "
        "response lines already transcribed above."
    ),
    "g0": (
        "G0 visual check: constants and equations are ordinary two-column tables. "
        "Fractions, superscripts, subscripts, and the periodic-table cell values are "
        "the same as the extracted text; no additional diagram or color code is used."
    ),
}

T1_TABLE = r"""
The compound catalogue is a 4-by-3 structure table.  Its numbered molecular
graphs are:
1: a benzene bearing adjacent OH and OCH3 groups and, para to OH, a
CH2-CH2-C(=O)-CH3 chain (formula C11H14O3).
2: a bicyclo[2.2.1]heptane skeleton bearing OH and three methyl substituents
(one bridgehead gem-dimethyl plus one methyl; C10H18O); no wedge/dash bonds are
printed.
3: an oxygen-bridged bicyclo[2.2.2]octane ether bearing one methyl and one
gem-dimethyl carbon (C10H18O); no wedge/dash bonds are printed.
4: a six-carbon unbranched chain with one internal C=C bond and terminal CH2OH
(C6H12O).
5: a benzene bearing adjacent OH and OCH3 groups and, para to OH, a
CH2-CH=CH2 group (C10H12O2).
6: the open chain CH2=CH-C(OH)(CH3)-CH2-CH2-CH=C(CH3)-CH3 (C10H18O).
7: an azulene graph (fused five- and seven-membered aromatic rings) bearing an
ethyl group and two methyl groups (C14H16).
8: the acyclic branched tetraene
CH2=CH-C(=CH2)-CH2-CH2-CH=C(CH3)-CH2-CH2-CH=C(CH3)2 (C15H24); no ring is
present.
9: a bicyclo[2.2.1]heptan-2-one framework bearing three methyl substituents
(C10H16O); no wedge/dash bonds are printed.
10: a cyclopentanone graph fused to a cyclopropane and bearing methyl and
isopropyl substituents (printed formula C10H18O); no wedge/dash bonds are
printed.
Rows map Zingiber to 1,2,3; Hypericum to 4,5,6; Chamomilla to 7,8,3; and
Artemisia to 9,10,3.
""".strip()

T1 = {
    "1.1": T1_TABLE + "\nThe answer sheet supplies independent check boxes 1-10 for X and for Y.",
    "1.2": T1_TABLE + "\nThe answer sheet supplies check boxes 1-10 for Z and two empty structure boxes labelled A and B.",
    "1.3": T1_TABLE + "\nThe answer sheet supplies a numeric line for n and check boxes 1-10 for W. The stated isotope information is textual; no spectrum trace is printed.",
    "1.4": "The upper reaction flow is Stone --HNO3--> an ellipsis (unknown dissolved species) --adjust to pH about 4, then NaF--> C*xH2O precipitate. A separate lower branch is C --NaF--> D. The answer sheet has text fields Q, C*xH2O, and D; no hidden structural scaffold.",
    "1.5": "The reaction sketch is E --KMnO4/HNO3--> F --P2O5--> G; no heat symbol or temperature is printed over the second arrow. E, F, and G are unknown blank nodes, not pre-drawn structures. Three independent drawing boxes are labelled E, F, and G.",
    "1.6": "No thermogravimetric trace is printed: the two mass-loss temperatures and masses are textual only. The answer sheet is blank working space followed by separate formula fields labelled Stone and H.",
}

T2_PHASE = (
    "The phase portrait has horizontal x=[Br-] and vertical y=[HBrO2]. It is a "
    "four-segment loop: upper-left ([Br-] below the critical value, y=A) to "
    "upper-right ([Br-]critical,y=A), diagonally to lower-right ([Br-]max,y=B), "
    "horizontally to lower-left ([Br-]critical,y=B), and diagonally back."
)
T2 = {
    "2.1": "The introductory time plot has two smooth periodic curves of equal period and roughly constant amplitude: solid blue [Ce3+] and dashed red [Ce4+] oscillate approximately half a period out of phase. The axes are concentration C versus time t and carry no numerical tick values. The answer sheet supplies blank working space only.",
    "2.2": "The mechanism is a ruled table. Processes A and B are separately braced groups of numbered equations (1)-(3) and (4)-(6); process C is equation (7). The reaction formulae and rate constants are fully present in the extracted text; braces add only the A/B grouping. The two-page answer area ends with separate numeric fields [HBrO2]A and [HBrO2]B, each in M.",
    "2.3": T2_PHASE + "\nThe answer sheet has blank working space and a final [Br-]critical field in M.",
    "2.4": T2_PHASE + "\nFour answer options form a 2x2 matrix. TL: upper horizontal left, lower horizontal right, left diagonal upper-left to lower-right, and right diagonal lower-right to upper-left. TR: upper horizontal right, lower horizontal left, left diagonal lower-right to upper-left, and right diagonal upper-left to lower-right. BL: both horizontals left and both diagonals upper-left to lower-right. BR: both horizontals right and both diagonals lower-right to upper-left. Each option has one check box.",
    "2.5": T2_PHASE + "\nThe answer sheet has blank working space and a final tau field in seconds.",
    "2.6": "The response grid has rows actions 1-4 and columns consequences a-e, with one check box at each intersection. The printed action and consequence wording above the grid is the complete option content; the grid contains no extra symbols.",
    "2.7": "Six curve choices form a 3x2 matrix for dG/dt versus t, with the horizontal axis also marking zero. TL: straight line decreasing from zero into negative values. TR: constant-amplitude sinusoid about zero. ML: negative exponential rising asymptotically to zero. MR: a small oscillation superposed on a downward negative trend. BL: oscillation about zero with growing amplitude. BR: negative damped oscillation approaching zero. One check box lies below each graph.",
}

T3_CATALOGUE = r"""
All molecular graphs in the monomer catalogue itself are black. In the
topology and reaction figures, black and red distinguish the two selected
building-block roles; they are not fixed colors for classes A-E.
A1 is a 1,3,5-triphenylbenzene with para-B(OH)2 on each outer phenyl
(3-connected); A2 is 1,4-benzenediboronic acid (2-connected); A3 is
tetra(4-boronophenyl)ethene (4-connected); A4 is tetra(4-boronophenyl)silane
(4-connected). B1 is a metal phthalocyanine-type square core with four
peripheral catechol pairs (4-connected); B2 is a naphthalene with two
catechol edges (four OH, 2-connected); B3 is hexahydroxytriphenylene (three
catechol edges, 3-connected). C1 is 4,4'-diaminobiphenyl (2-connected); C2
has a central diarylethene-like pair of benzene nodes and four para-
aminophenyl arms (4-connected); C3 is a metal tetrakis(4-aminophenyl)-
porphyrin (4-connected). D1 is hexakis(4-formylphenyl)benzene (6-connected);
D2 is 2,4,6-tris(4-formylphenyl)-1,3,5-triazine (3-connected); D3 is
tetrakis(4-formylphenyl)methane (4-connected); D4 is
2,3,5,6-tetrafluoro-1,4-benzenedicarbaldehyde (2-connected). E1 is a
1,3,5-tris(cyanomethyl)benzene (three -CH2-CN arms, 3-connected).
The seven topology icons preserve colored building-block roles: tetragonal-1
is 4-connected black nodes joined by 2-connected red linkers; tetragonal-2
interchanges those color/connectivity roles; hexagonal-1 is 3-connected black
nodes plus 2-connected red linkers; hexagonal-2 alternates 3-connected black
and 3-connected red nodes; trigonal uses 6-connected black nodes and
2-connected red linkers; kagome uses 4-connected black nodes and 2-connected
red linkers; tetrahedral is the three-dimensional diamond net of 4-connected
black nodes and 2-connected red linkers.
""".strip()

T3 = {
    "3.1": "COF-1 is drawn as a two-dimensional honeycomb boronate-ester network. Each large hexagonal pore is bounded by alternating triangular hexahydroxytriphenylene-derived nodes and para-phenylene diboronate linkers. Short blue dashed cut lines mark the illustrated repeat-unit edges; they do not form a red polygon. The source monomer/reaction drawing uses B3 plus A2 with loss of water. No stereobonds occur. The answer sheet has a COF-1 empirical-formula field and a separate carbon-percentage field.",
    "3.2": "COF-2 is a honeycomb boroxine network formed by self-condensation of A2: B3O3 rings occupy the corners and para-phenylene groups lie along the pore edges. The internal diameter d is the straight distance across opposite sides of the regular hexagonal pore. The listed C-C/C=C, C-B, and B-O lengths and the inscribed-hexagon relation supply the geometry; no second colored monomer is present. The answer sheet has a single d field in angstroms.",
    "3.3": T3_CATALOGUE + "\nThe answer table columns are tetragonal-1, tetragonal-2, hexagonal-1, hexagonal-2, trigonal, kagome, tetrahedral. Prefilled examples are A2+B3 under hexagonal-1, E1+D2 under hexagonal-2, and C2+D4 under kagome; two blank rows remain in every topology column.",
    "3.4": T3_CATALOGUE + "\nCOF-3 is made from benzene-1,3,5-tricarbaldehyde plus 2,5-dihydroxybenzene-1,4-diamine, then oxidized to COF-4. COF-5 is made from 2,4,6-trihydroxybenzene-1,3,5-tricarbaldehyde plus p-phenylenediamine, then oxidized to COF-6. Blue dashed cut lines in the COF-1 example illustrate the required repeat-edge convention. Four blank drawing panels are labelled COF-3 through COF-6.",
    "3.5": "The scheme starts with C2 and D4 (graphs defined in the catalogue), AcOH, and a kagome-like COF-7 sheet; sequential NaClO2/NaH2PO4 and then O3/Me2S produce macrocycle X after degradation. The COF-7 topology fragment uses black 4-connected nodes and red 2-connected linkers; no dashed repeat boundary and no atoms of unknown X are prefilled. The answer sheet is one empty drawing box.",
    "3.6": "Three stacking cartoons are labelled AA (exactly eclipsed pores), AA' (laterally slipped layers), and AB (alternating offset layers). The interaction-energy table has eclipsed/slipped columns and rows benzene-benzene (b-b: -7.9/-12.6), benzene-triazine (b-t: -49.8/-55.6), and triazine-triazine (t-t: -6.7/-16.7), in kJ mol-1. COF-8 is drawn from D2 and E1 as a hexagonal cyano-vinylene/C=C-linked pore, not an imine-linked pore. The answer table asks AA, AA', AB, and AB' and preprints AA'=-67.1 kJ mol-1.",
    "3.7": "COF-8 and COF-9 are the same hexagonal framework before and after conversion of red nitrile groups into red amidoxime groups by NH2OH. The small photograph shows a blue fluorescent cuvette before UO2(2+) addition and a dark/non-fluorescent cuvette after addition; it carries no numerical scale. The pore diagram shows the red functional groups at the three-connected nodes and black aromatic linkers around a hexagonal opening. The answer sheet has separate fields for q_e in mg g^-1 and the COF-9:UO2(2+) ratio.",
}

T4 = {
    "4.1": "No diagram is printed. The answer sheet is blank working space followed by a 235U percentage field.",
    "4.2": "No diagram is printed beyond the nuclear-isotope notation in the text. The answer sheet provides three separate equation lines labelled (a), (b), and (c).",
    "4.3": "The fission-yield graph has mass number A from 70 to 170 on the horizontal axis and P(A) from 0 to 8 percent on the vertical. A smooth blue double-humped curve has nearly equal maxima P about 7.0 at A about 93 and A about 140, a minimum near P about 0.1 at A about 116, and approaches zero near A=70 and 170. Major x ticks are every 10 mass units and major y ticks every 0.5 percentage point; the answer sheet is an unseeded working box.",
    "4.4": "The same double-humped fission-yield graph from 4.3 remains contextual. The answer sheet is blank working space followed by a Delta E field in MeV.",
    "4.5": "No additional graph is printed; the logarithmic-decrement equation and numerical inputs are textual. The answer sheet is blank working space followed by an n_c field.",
    "4.6": "The thermodynamic data are a two-column table: Delta_f H298(CH4)=-74.8 kJ mol^-1, Delta_f H298(H2O,gas)=-241.8 kJ mol^-1, Delta_f H298(CO2)=-393.5 kJ mol^-1; C_P(CH4)=35, C_P(H2O,gas)=34, C_P(O2)=29, and C_P(CO2)=37 J mol^-1 K^-1. The answer sheet ends with a Delta_r H298 field in kJ mol^-1.",
    "4.7": "The same thermodynamic table supplies the aligned values. In red question text, the fallback for a missing 4.6 result is Delta_r H298=-750 kJ mol^-1, and the fallback printed after 4.7 for later calculations is Delta_r H2000=-700 kJ mol^-1; these are question material, not marking material. The answer sheet has a Delta_r H2000 field in kJ mol^-1.",
    "4.8": "No graph is printed. The red question-text fallback immediately above is Delta_r H2000=-700 kJ mol^-1 if 4.7 was not obtained. The answer sheet has an E field in J day^-1.",
    "4.9": "The fission-yield graph from 4.3/4.4 is contextual. A red question-text fallback gives Delta E=200 MeV if 4.4 was not obtained. The answer sheet is blank working space followed by separate TN and mass m (kg) fields.",
}

T5_FRAGMENTS = r"""
The fragment legend uses a wavy bond as a connection stub. a is a terminal
H cap (~-H). b is a phosphoric-acid connector ~-P(=O)(OH)-~ with two stubs.
c is a glycerol tri-oxygen connector ~O-CH2-CH(O~)-CH2-O~ with three stubs.
d is an acyl connector ~-C(=O)R. e is ~-OH and appears only in the worked
example. The example W connects one of a-e as
HO-P(=O)(OH)-O-CH2-CH(OH)-CH2-O-C(=O)R; the central glycerol carbon is drawn
with its OH on a solid wedge toward the viewer. In the actual PL1 palette,
the multiplicities are n*a, 2*b, 3*c, and 4*d.
""".strip()

T5 = {
    "5.1": T5_FRAGMENTS + "\nThe response is a three-column choice table headed (a),(b),(c), with one blank lower cell under each heading; no small check-box glyphs are printed.",
    "5.2": T5_FRAGMENTS + "\nThe prompt explicitly requires the enantiomer of PL1 and a hydrogen-bonded form Y; the answer page has two unseeded drawing boxes labelled PL1 and Y. No answer-bearing bonds are preprinted in either box.",
    "5.3": T5_FRAGMENTS + "\nThe relevant species is the solver's PL1; no fatty-acid chain structure is prefilled. The answer sheet has lines for the double-bond count and RCOOH formula.",
    "5.4": "The iodine-addition relation is textual. The answer area is an unseeded formula line X; no structural hint is printed.",
    "5.5": "Four phase cartoons are labelled: (a) normal spherical micelle, polar heads on the outside and hydrocarbon tails inward; (b) stacked planar bilayers, headgroups at each water-facing surface and tails meeting in each slab; (c) normal hexagonal cylinders, headgroups on cylinder exteriors and tails toward the axes; (d) inverse hexagonal cylinders, headgroups/water at cylinder cores and tails in the continuous exterior. The response is a four-column choice table headed a-d, with one blank lower cell per choice; no small check-box glyphs are printed.",
    "5.6": T5_FRAGMENTS + "\nThe hydrolysis drawings correspond to PL1+8H2O -> 4RCOOH+2H3PO4+3 glycerol; PL2+8H2O -> 4RCOOH+2H3PO4+Z+2 glycerol; and PL3+mH2O -> 2RCOOH+H3PO4+ethanol+glycerol. The displayed glycerol is HOCH2-CH(OH)-CH2OH. Answer pages provide empty boxes PL2 and PL3 only; no product bonds are seeded.",
}

T6 = {
    "6.1": "The response table has rows A and AA and columns C18, C16, triplet-C13, singlet-C13. No cells are prefilled, and no molecular line structure is printed inside the 6.1 response area.",
    "6.2": "The starting graph is three linearly fused six-membered rings (an anthracene framework) whose ten perimeter sites are chlorinated. The sequence is starting material -2Cl radical -> A; A -5Cl radical -> B; B -2Cl radical -> C; C -Cl radical -> C14; and A -3Cl radical -> D. AFM panels A-D are grayscale bond-resolved images: A shows three fused-ring lobes with remaining peripheral bright Cl spots; B a shorter bent fused-carbon pattern; C an annular two-ring-intensity image; D an elongated zigzag carbon skeleton; C14 is a single circular ring image. The answer sheet preprints A as an anthracene graph bearing eight perimeter Cl atoms and two radical dots, at the upper non-fused carbon of the right ring and the lower non-fused carbon of the central ring; boxes B-D are blank.",
    "6.3": "The bond-dissociation table is C-F 467, C-Cl 346, C-Br 290, C-I 228 kJ mol-1. The answer sheet has one check/entry line X; no extra drawing.",
    "6.4": "Two separate structures are drawn side by side: C48 as a large circular cyclocarbon ring with repeated multiple-bond strokes, and macrocycle E as a roughly C2-symmetric aromatic/ether macrocycle containing an upper bipyridyl N2 unit, two side ether O atoms, and one bottom ether O, formula C40H34N2O3. The source text states that they form interlocking rings, but the figure does not draw an actual crossing or interlocked composite. The mass table has m/z 591 (example, prefilled ion [E+H]+), 783, 879, 1174 with blank ion assignments.",
    "6.5": "The synthesis starts from 4-(4-bromophenyl)-4-hydroxycyclohexa-2,5-dien-1-one. Step 1 NaH; step 2 uses 4-lithio-4'-(OTBS)biphenyl to give F. F +2 equiv TESCl/imidazole -> G; G +LiOH -> H (C36H47BrO3Si2); H +PhI(OAc)2/H2O -> I. I then undergoes NaH and p-bromophenyllithium -> J; J +2 equiv TESCl -> K; K +2 equiv Ni(COD)2/2,2'-bipyridine -> L (C54H80O4Si4); L +4 equiv TBAF then excess SnCl2 -> [5]CPP. TBS is tert-butyldimethylsilyl, TES triethylsilyl, COD cycloocta-1,5-diene. Answer boxes F-L are completely blank and do not seed product connectivity.",
    "6.6": "The scheme begins with toluene and 2 equiv tert-butyl chloride/AlCl3 to M (thermodynamic product; four proton environments). M undergoes NBS/benzoyl peroxide, then HMTA, then HCl/H2O to N. N plus dipyrromethane/CF3CO2H, then DDQ, then Zn(OAc)2 gives porphyrin O; NBS gives Q; Pd(0)/PPh3/CuI with triisopropylsilylacetylene, then TBAF gives R. Six R units couple oxidatively around the depicted hexapyridylbenzene template under p-benzoquinone/PdCl2/PPh3/CuI, and DABCO removes the template to give the six-porphyrin belt P6. The belt drawing shows six Zn-porphyrins around a cylindrical rim joined by diyne edges; answer boxes M,N,O,Q,R are blank.",
    "6.7": "P6 is a cyclic belt of six Zn-porphyrin units connected by twelve alkyne-derived links. The global aromatic pathway is the bold perimeter path only; the example [n]CPP diagram is n para-linked benzene rings and states that only its bold circumferential path is counted. Oxidation removes electrons from P6; no answer count is printed. The answer sheet has lines n(e) and n(t).",
}

T7_FLOW = "The plant flow is: xCH4+yH2O enters steam reformer CH4+H2O->CO+3H2; that stream (CH4,CO,H2) enters partial oxidation 2CH4+O2->2CO+4H2 together with 4N2+1O2, yielding M1 downward to water-gas shift CO+H2O->CO2+H2 with H2O feed. The N2/CO2/H2 stream flows left through CO2 scrubber Z; N2/H2 go down to N2+3H2<=>2NH3. M2 flows right to cooler CLR; NH3 exits right, while unreacted N2/H2 recycle left to the reactor. Red arrows establish these directions."

T7 = {
    "7.1": T7_FLOW + "\nThe answer table has gas columns CH4,H2O,CO,O2,H2,N2,NH3,CO2 for rows M1/M2, plus three relationship choices x>y, x=y, x<y.",
    "7.2": T7_FLOW + "\nThe answer page is blank working space with a final m(CH4) line.",
    "7.3": "The recycle diagram has N2+3H2 feed into the ammonia reactor, M2 flowing right to cooler CLR, NH3 leaving right, and unreacted N2/H2 returning left in a lower loop. The arrow directions and species exactly match the four textual cycle steps. Answer space includes lines f_n and n plus a separate 7.4 box.",
    "7.4": "Compound 3 is N-methyldiethanolamine, HO-CH2-CH2-N(CH3)-CH2-CH2-OH. The answer space is an unseeded equation box.",
    "7.5": "Complex 4 is MoCl3 bound by a tridentate PNP pincer: a central pyridine N and two CH2-P(tBu)2 arms; the simplified ligand icon is an N-P-P arc. N2 at 1.0 bar and Na-Hg (3 equiv) lead to unknown 5. Mw(4)=597.824 g mol-1. No connectivity of 5 is preprinted; the answer has a blank drawing box.",
    "7.6": "Reference table 1-4: reductants are chromocene for 1 (E°=-0.88 V) and cobaltocene for 2-4 (E°=-1.15 V). Additive 1 is 2,6-dimethylpyridinium triflate (pKa14.4); 2 is 2,6-dimethylpyridinium chloride (14.4); 3 is 2-methylpyridinium triflate (13.9); 4 is 2,6-dimethylpyridinium triflate (14.4). NH3 yields are 0,0.7,9.1,11.8 mol. Candidate table A-D: A nickelocene, E°=-0.09, phenanthrolinium triflate pKa13.7; B vanadocene, -1.10, 2,4,6-trimethylpyridinium triflate pKa15.0; C vanadocene, -1.10, phenanthrolinium triflate pKa13.7; D vanadocene, -1.10, anilinium triflate pKa10.6.",
    "7.7": "All composition clues are textual. The response sheet provides separate formula lines 7,8,9,10 and separate S and T lines; no lattice drawing or elemental color key is supplied.",
}

T8 = {
    "8.1": "The only diagram is CO2 --catalyst 1--> CO; catalyst 1 is an unknown label at this point. The response box is unseeded.",
    "8.2": "Sacrificial reductant 2 is triethanolamine N(CH2CH2OH)3. The drawn sequence is 2 -1e- -> [3] -H+ -> [4] -1e- -> [5] then +H2O/-H+ -> 6+7; square brackets and radical/charge notation are to be supplied by the solver. Five independent drawing panels 3-7 are blank.",
    "8.3": "Ligand 8 is a four-pyridyl tetradentate chain: a carboxyphenyl-substituted terminal pyridine joined through its 6-position to three further 2,2'-linked pyridines; a cartoon shows four N donors joined by three downward arcs. Seven geometry options are arranged 4 above/3 below: four-coordinate square-planar 4N; four-coordinate tetrahedral 4N; 4N plus one axial Cl; 4N plus one axial Cl in an alternative folded ligand orientation; octahedral 4N+2Cl; octahedral-like 4N+2Cl with one Cl equatorial; and 4N+2 trans axial Cl. Each has one check box.",
    "8.4": "The catalytic cycle is clockwise: 1 +2H2O/-2Cl- ->9; 9 +hν+1e- -H2O ->10; 10 -H2O/+CO2 ->11; 11 +hν+1e-+H+ ->12; 12 +H2O ->13; 13 +H+/-H2O ->14; 14 -CO ->15; 15 +H2O ->9. Answer templates 9-15 each have a large drawing box and fields Z, OS, CN, VE. The only prefilled descriptor fields are OS=+3 and CN=6 for 11, CN=5 for 12, OS=+2 for 13, and VE=16 for 15; all other descriptor and charge fields are blank.",
    "8.5": "The catalyst-cycle figure from page 2 is contextual only; the numerical surface-area task and its answer line Ncat contain no additional graph.",
    "8.6": "The cycle figure is contextual. The formula phi=(number of reacted electrons)/(number of incident photons)*100% is displayed as a fraction; the answer sheet has phi=___%. No omitted visual value occurs.",
    "8.7": "TOF data plot: x=omega_cat percent and y=CO TOF h^-1. Red points connected by a dashed curve are exactly (0.1,62),(0.3,56),(0.6,37),(1.0,29),(2.0,15),(2.9,11). Vertical dotted guides connect each point to x. Choices a increases, b decreases, c does not change appear as check boxes.",
    "8.8": "Free-energy diagram has two paths from ΔG=0. Upper H2 path step labels, in order, are +H+ with +0.756 eV; +e- with +1.832 eV; +H+ with -0.168 eV; and -H2 with -1.14 eV. Lower CO path: +e- -0.04 eV; +CO2 +0.02 eV; +e- +0.16 eV; +H+ -0.36 eV; +H+,-H2O -0.22 eV; -CO +0.10 eV. Stacked bars a-d: a has no bar (zero products); b totals100% with H2 about28% and CO about72%; c H2 about8%, CO about92%; d H2 about4%, CO about96%. Blue diagonal hatch=H2, red vertical hatch=CO. Answer matrix maps each of a-d to N,R,G,B.",
    "8.9": "Jablonski diagram: S0 is the bottom manifold; S1 is upper-left and T1 upper-right. Downward S1->S0 arrows are red fluorescence kF and black internal conversion kIC; downward T1->S0 arrows are blue phosphorescence kP and black nonradiative kN. Curved S1->T1 is kISC and T1->S1 is kRISC. Outer quenching paths from Red at S1 and T1 to Red radical-cation are kS and kT. No arrow magnitude is encoded by length or width. The answer sheet ends with separate percentage lines eta_q(S1) and eta_q(T1).",
    "8.10": "The same Jablonski diagram and arrow mapping as 8.9 applies. The answer sheet offers a,b,c only; no state-specific separate choice is printed.",
}

T9_TEMPLATE = "The CD template is a top-view annulus. Seven (beta-CD) or six (alpha-CD) primary substituent boxes lie around the inner rim in cyclic order; a single outer box labelled (OH)14 for beta-CD or (OBn)12 for alpha-CD denotes the repeated secondary substituents. Red position numbers increase clockwise where printed."

T9 = {
    "9.1": "The figures show an alpha-D-glucopyranose repeat in a 4C1 chair joined through alpha-1,4 glycosidic O bonds, a truncated-cone CD cartoon, a top-view seven-box beta-CD template, and a full ring labelled n=7 beta-CD. In that full-ring source graphic, one right-hand substituent is visibly labelled OBn even though the surrounding hydroxyls and caption identify beta-CD; this is a source-page anomaly, not a supplied answer. The molar-mass data are textual, and the answer sheet has one M_w: beta-CD line.",
    "9.2": "The central repeat is beta-CD (seven alpha-D-glucopyranose chairs). Left synthesis: NaIO4; NaBH4/H2O; Ac2O/pyridine ->X. Right: TsCl (7 equiv)/pyridine; NaOH/H2O,60°C ->K. The answer sheet gives two seven-repeat CD chair templates as the two choices: left, standard 4C1 glucopyranose chair with ring O upper-right and alpha glycosidic O axial-down at C1; right, inverted 1C4 chair with the ring O in the upper edge and the outgoing glycosidic O equatorial. Each template has one check box; the chosen template itself is to be completed as K, and there is no separate third K template or seeded substituent.",
    "9.3": "The same beta-CD repeat and left-hand NaIO4/NaBH4/Ac2O route to X are shown. The macrocyclic loop is drawn around all seven repeat units; there is no numerical ring-size annotation. Answer fields are rs and sc only.",
    "9.4": "Alpha-cycloaltrin route: alpha-CD (six standard glucopyranose chairs) treated with TBSCl (6 equiv)/pyridine, then NaH (12 equiv), TsCl (6 equiv) gives Y, formula C72H132O24Si6; t-Bu4NF then H2O,100°C gives the displayed alpha-cycloaltrin repeat, in which each glucose-derived six-membered ring has ring O, one OH up, one OH down, and an additional O connection consistent with the new cyclic acetal/ether framework. TBSCl is tBuMe2Si-Cl. The answer template is a six-unit chair loop with no substituents prefilled.",
    "9.5": T9_TEMPLATE + "\nStarting beta-CD positions 1-7 are printed clockwise with CH2OH in every primary box and (OH)14 outside. NaH (30 equiv), BnCl (30 equiv), then DIBAL-H (2 equiv) gives L. The later displayed dimer has two beta-CD rims joined by an -O-(CH2)8-O- bridge between primary CH2 groups; on each ring five primary boxes are CH2OBn, one is CH2OH, and one bridge-bearing box is CH2O-(CH2)8-OCH2; all fourteen secondary sites are OBn. The requested L answer template has seven blank primary boxes and one outside secondary-substituent box.",
    "9.6": T9_TEMPLATE + "\nThe final dimer drawing contains two seven-position rings joined through the same -O-(CH2)8-O- primary-rim bridge described for 9.5. Red position numbers 1-7 establish cyclic adjacency and the unit-4/unit-3 directing rule; no list of isomers is printed.",
    "9.7": "The degradation substrate fragment shows three consecutive glucopyranose units of L: every shown secondary OH is OBn, primary sites on the outer units are OBn, and the middle unit has a free primary OH. Reagents I2/P(C6H5)3; Zn/C3H7OH; NaBH4; Ac2O/pyridine cleave it into products whose printed neutral formulae are C27H28O5 and C22H25O4. The structures show acetylated unsaturated sugar fragments, but the formulae are the authoritative mass inputs. Answer lines are [M1+Na]+ and [M2+Na]+.",
    "9.8": T9_TEMPLATE + "\nStarting alpha-CD N has position1=CH2OH and positions2-6=CH2OBn, with (OBn)12 outside. N --(COCl)2,DMSO; Ph3P=CH2--> O --DIBAL-H(1); MsCl; NaN3--> P --DIBAL-H(2); NaH plus ClCH2-C(=CH2)-CH2Cl--> Q --DIBAL-H(1); Boc2O(1); MsCl; NaN3--> R --CF3CO2H; NaH/BnI; DIBAL-H(2)--> S. The branching alkyl reagent is 3-chloro-2-(chloromethyl)prop-1-ene. Answer sheets show four six-position annuli O,P,Q,R and one S, with red positions1-6 and an outside OBn12 box; every substituent box is initially blank.",
    "9.9": "The response sheet is blank working space. The question concerns cyclic arrangements of six modified CH2OH sites; there is no colored arrangement diagram or prefilled permutation in the official question/answer pages.",
}

ALL = {}
for problem in (T1, T2, T3, T4, T5, T6, T7, T8, T9):
    ALL.update(problem)

assert len(ALL) == 68, len(ALL)

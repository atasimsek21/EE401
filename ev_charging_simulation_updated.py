"""
===========================================
SMART EV FAST CHARGING SIMULATION - UPDATED
===========================================
Interactive simulation for 2-EV charging with 8 allocation scenarios
Features:
- Realistic Power vs SOC curves for each EV model
- Heuristic scenarios: Equal Share, FIFO, SoC-Based, Energy-Needed, Deadline+Energy, FIFO+Energy
- MILP scenarios: Deadline-Based, Max-Min Fairness
- Deadline feasibility checking with user guidance
- Comprehensive plotting and performance metrics
- UPDATED: Standardized allocation flow with binding constraint phase detection
"""

import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, LpMinimize, lpSum, PULP_CBC_CMD, value as lp_value
import sys

# Try to import and test optional solvers
SOLVER_AVAILABLE = {"CBC": True, "HiGHS": False, "Gurobi": False}
try:
    from pulp import HiGHS_CMD
    import subprocess
    subprocess.run(['highs', '--version'], capture_output=True, timeout=1)
    SOLVER_AVAILABLE["HiGHS"] = True
except:
    pass
try:
    from pulp import GUROBI_CMD
    import subprocess
    subprocess.run(['gurobi_cl', '--version'], capture_output=True, timeout=1)
    SOLVER_AVAILABLE["Gurobi"] = True
except:
    pass

# ============================================================================
# 1. DATA DEFINITIONS
# ============================================================================

EVS = {
    "Tesla Model 3 SR (LFP 64)": {
        "chem": "LFP", "capacity_kWh": 64.0, "nominal_voltage": 339,
        "P_curve": [(0.05, 40), (0.10, 130), (0.20, 95), (0.30, 85), (0.40, 70),
                    (0.50, 60), (0.60, 45), (0.70, 35), (0.80, 28), (0.90, 18),
                    (0.98, 8), (1.00, 0)]
    },
    "BYD Seal AWD (Blade 85.4)": {
        "chem": "LFP", "capacity_kWh": 85.4, "nominal_voltage": 569,
        "P_curve": [(0.05, 50), (0.10, 120), (0.12, 150), (0.20, 150), (0.30, 148),
                    (0.40, 145), (0.45, 145), (0.48, 100), (0.55, 85), (0.65, 70),
                    (0.72, 55), (0.80, 70), (0.85, 70), (0.88, 45), (0.95, 45),
                    (0.98, 10), (1.00, 0)]
    },
    "Kia EV6 LR 2WD (84)": {
        "chem": "NMC", "capacity_kWh": 84.0, "nominal_voltage": 697,
        "P_curve": [(0.05, 220), (0.10, 235), (0.20, 245), (0.30, 255), (0.45, 265),
                    (0.55, 270), (0.60, 240), (0.65, 220), (0.70, 160), (0.75, 130),
                    (0.80, 90), (0.90, 40), (0.98, 8), (1.00, 0)]
    },
    "Porsche Taycan 4 (105 PBP)": {
        "chem": "NMC", "capacity_kWh": 105.0, "nominal_voltage": 723,
        "P_curve": [(0.05, 250), (0.10, 270), (0.20, 285), (0.30, 295), (0.40, 290),
                    (0.55, 285), (0.60, 270), (0.65, 200), (0.70, 210),
                    (0.75, 150), (0.80, 110), (0.90, 60), (0.95, 35), (0.99, 5), (1.00, 0)]
    }
}

STATIONS = {
    "150kW Fast Charger": {"P_total_kW": 150, "modules": 5, "module_kW": 30,
                          "Vmin": 150, "Vmax": 1000, "Imax": 500},
    "ABB Terra HP 350kW": {"P_total_kW": 350, "modules": 10, "module_kW": 35,
                          "Vmin": 150, "Vmax": 920, "Imax": 500},
    "Delta UFC 500kW": {"P_total_kW": 500, "modules": 10, "module_kW": 50,
                       "Vmin": 200, "Vmax": 920, "Imax": 500},
}

OCV = {
    "LFP": {"soc": [0.0, 0.1, 0.4, 0.8, 0.95, 1.0],
            "Vcell": [2.8, 3.0, 3.15, 3.30, 3.40, 3.45]},
    "NMC": {"soc": [0.0, 0.1, 0.5, 0.8, 0.95, 1.0],
            "Vcell": [3.0, 3.4, 3.60, 3.90, 4.10, 4.20]}
}

SCENARIOS = [
    "equal_share", "fifo", "soc_based", "deadline",
    "energy_needed", "deadline_energy", "fifo_energy", "max_min_fairness"
]

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def interp1(x_points, y_points, x):
    """Linear interpolation"""
    return np.interp(x, x_points, y_points)

def pack_voltage(ev, soc):
    """Calculate battery pack voltage at given SOC"""
    chem = ev["chem"]
    ocv_data = OCV[chem]
    vcell = interp1(ocv_data["soc"], ocv_data["Vcell"], soc)

    # Estimate cell count from nominal voltage
    vcell_nominal = 3.2 if chem == "LFP" else 3.7
    cell_count = int(round(ev["nominal_voltage"] / vcell_nominal))

    return vcell * cell_count

def pmax_curve(ev, soc):
    """Get maximum power from EV's power curve at given SOC"""
    soc_points = [p[0] for p in ev["P_curve"]]
    power_points = [p[1] for p in ev["P_curve"]]
    return interp1(soc_points, power_points, soc)

def p_cap_by_modules(m, station):
    """Power capacity in kW based on allocated modules"""
    return m * station["module_kW"]

def i_cap_by_modules(m, V, station):
    """Current capacity in A based on allocated modules and voltage"""
    if V <= 0:
        return 0.0
    P_mod_cap = m * station["module_kW"] * 1000  # W
    return P_mod_cap / V  # A

def enforce_total_power(P1, P2, station):
    """
    Enforce total power constraint by proportional scaling if needed
    Returns: (P1_adjusted, P2_adjusted)
    """
    Psum = P1 + P2
    if Psum <= station["P_total_kW"] + 1e-9:  # Small tolerance
        return P1, P2
    
    # Proportional scaling (orantılı kısıtlama)
    scale = station["P_total_kW"] / Psum
    return P1 * scale, P2 * scale

def classify_phase(Pdel, I, V, P_curve, Imax, Pmod_kW, Pshare_kW):
    """
    Classify charging phase based on binding constraint approach
    Returns: (phase_str, caps_dict)
    
    Phase determination:
    - CC if delivered power ≥ 95% of the minimum limiting factor
    - CV otherwise
    """
    if Pdel <= 0.01:  # Not charging
        return "Done", {}
    
    # Calculate all capacity limits (in kW)
    caps = {
        "curve": P_curve,
        "current": (Imax * V) / 1000 if V > 0 else float('inf'),
        "modules": Pmod_kW,
        "share": Pshare_kW
    }
    
    # Find minimum capacity (binding constraint)
    P_cap = min(caps.values())
    
    # CC if delivered power is close to capacity (≥95%)
    if Pdel >= 0.95 * P_cap:
        phase = "CC"
    else:
        phase = "CV"
    
    return phase, caps

def energy_update(soc, I, V, capacity_kWh, eta=0.95, dt_min=1):
    """Update SOC after charging for dt_min with current I and voltage V"""
    energy_added_kWh = (I * V * (dt_min/60) * eta) / 1000.0
    new_soc = soc + energy_added_kWh / capacity_kWh
    return min(1.0, new_soc), energy_added_kWh

def estimate_charging_time(ev, soc_start, soc_target, station, verbose=False):
    """Estimate minimum charging time assuming EV gets full station power"""
    soc = soc_start
    time = 0
    max_iter = 500  # Safety limit

    while soc < soc_target and time < max_iter:
        v = pack_voltage(ev, soc)
        if v < station["Vmin"] or v > station["Vmax"]:
            if verbose:
                print(f"  ⚠️  Voltage out of range at SOC={soc*100:.1f}%")
            return float('inf')

        p_max = pmax_curve(ev, soc)
        p_available = min(p_max, station["P_total_kW"])
        i = min((p_available * 1000) / v, station["Imax"])

        soc, _ = energy_update(soc, i, v, ev["capacity_kWh"])
        time += 1

    return time

def check_deadline_feasibility(ev1, ev2, station, soc1_init, soc2_init,
                               soc1_target, soc2_target, arrival2,
                               deadline1, deadline2, verbose=True):
    """Check if deadlines are achievable and provide guidance"""

    if verbose:
        print("\n" + "="*60)
        print("DEADLINE FEASIBILITY CHECK")
        print("="*60)

    # Estimate time for EV1 alone
    time_ev1_alone = estimate_charging_time(ev1, soc1_init, soc1_target, station, verbose)

    # Estimate time for EV2 alone
    time_ev2_alone = estimate_charging_time(ev2, soc2_init, soc2_target, station, verbose)

    if verbose:
        print(f"\nEV1 minimum charging time (alone): {time_ev1_alone:.1f} minutes")
        print(f"EV2 minimum charging time (alone): {time_ev2_alone:.1f} minutes")
        print(f"EV2 arrives at: {arrival2} minutes")
        print(f"\nUser-specified deadlines:")
        print(f"  EV1: {deadline1} minutes")
        print(f"  EV2: {deadline2} minutes (from t=0)")

    feasible = True
    messages = []

    # Check EV1
    if deadline1 < time_ev1_alone:
        feasible = False
        messages.append(f"❌ EV1 deadline ({deadline1} min) is too short!")
        messages.append(f"   Minimum required: {int(np.ceil(time_ev1_alone))} minutes")
        messages.append(f"   Recommended: {int(np.ceil(time_ev1_alone * 1.2))} minutes (with 20% buffer)")

    # Check EV2
    time_available_ev2 = deadline2 - arrival2
    if time_available_ev2 < 0:
        feasible = False
        messages.append(f"❌ EV2 deadline ({deadline2} min) is before its arrival ({arrival2} min)!")
        messages.append(f"   EV2 needs at least {int(np.ceil(time_ev2_alone))} minutes after arrival")
        messages.append(f"   Minimum deadline: {arrival2 + int(np.ceil(time_ev2_alone))} minutes")
    elif time_available_ev2 < time_ev2_alone:
        feasible = False
        messages.append(f"❌ EV2 deadline gives only {time_available_ev2:.1f} minutes to charge!")
        messages.append(f"   Minimum required: {int(np.ceil(time_ev2_alone))} minutes")
        messages.append(f"   Recommended deadline: {arrival2 + int(np.ceil(time_ev2_alone * 1.2))} minutes")

    if verbose:
        print("\n" + "-"*60)
        if feasible:
            print("✅ FEASIBLE: Deadlines are achievable")
        else:
            print("❌ INFEASIBLE: Deadlines cannot be met")
            print("\nIssues found:")
            for msg in messages:
                print(msg)
        print("-"*60)

    return feasible, time_ev1_alone, time_ev2_alone, messages

# ============================================================================
# 3. SOLVER SELECTION
# ============================================================================

def get_solver():
    """Select best available solver"""
    if SOLVER_AVAILABLE["Gurobi"]:
        from pulp import GUROBI_CMD
        print("Using Gurobi solver (fastest)")
        return GUROBI_CMD(msg=0)
    elif SOLVER_AVAILABLE["HiGHS"]:
        from pulp import HiGHS_CMD
        print("Using HiGHS solver (fast)")
        return HiGHS_CMD(msg=0)
    else:
        print("Using CBC solver (default)")
        return PULP_CBC_CMD(msg=0)

# ============================================================================
# 4. POWER ALLOCATION FUNCTIONS (STANDARDIZED FLOW)
# ============================================================================

def allocate_power(ev1, ev2, station, soc1, soc2, t, arrival2, target1, target2,
                   scenario, deadline1, deadline2):
    """
    Allocate power between EV1 and EV2 using STANDARDIZED FLOW
    
    Returns: (I1, I2, m1, m2, phase1, phase2)
    
    Standardized Flow:
    1. Determine active EVs
    2. Calculate V1, V2, P_curve1, P_curve2
    3. Minimum guarantee modules (m_min = 1 if both active)
    4. Scenario-specific m_free distribution and Pshare ceilings
    5. P_raw = min(P_curve, Pmod, Imax*V/1000, Pshare)
    6. enforce_total_power(P1_raw, P2_raw)
    7. I = P*1000/V
    8. classify_phase for CC/CV determination
    9. Return results
    """
    
    # ====================
    # STEP 0: Active EVs
    # ====================
    ev1_active = (soc1 < target1)
    ev2_active = (t >= arrival2) and (soc2 < target2)
    
    # ====================
    # STEP 1: Voltage & Curve
    # ====================
    V1 = pack_voltage(ev1, soc1) if ev1_active else pack_voltage(ev1, soc1)
    V2 = pack_voltage(ev2, soc2) if (t >= arrival2) else np.nan
    
    Pcurve1 = pmax_curve(ev1, soc1) if ev1_active else 0.0
    Pcurve2 = pmax_curve(ev2, soc2) if ev2_active else 0.0
    
    # ====================
    # STEP 2: Minimum Guarantee Modules
    # ====================
    both_active = ev1_active and ev2_active
    
    if not ev1_active and not ev2_active:
        # No active EVs
        return 0.0, 0.0, 0, 0, "Done", "Done"
    
    elif both_active:
        # Both active: minimum guarantee
        m_min = 1
        m1 = m_min
        m2 = m_min
        m_free = max(0, station["modules"] - m1 - m2)
    else:
        # Only one active
        m_free = 0
        if ev1_active:
            m1 = station["modules"]
            m2 = 0
        else:
            m1 = 0
            m2 = station["modules"]
    
    # ====================
    # STEP 3: SCENARIO-SPECIFIC MODULE DISTRIBUTION & POWER SHARE
    # ====================
    Pshare1 = float('inf')  # Default: no share limit
    Pshare2 = float('inf')
    
    if scenario == "equal_share":
        if both_active:
            m1 += m_free // 2
            m2 = station["modules"] - m1
            Pshare1 = station["P_total_kW"] / 2
            Pshare2 = station["P_total_kW"] / 2
    
    elif scenario == "soc_based":
        if both_active:
            w1 = max(0.01, 1.0 - soc1)
            w2 = max(0.01, 1.0 - soc2)
            total_w = w1 + w2
            
            m1 += round(m_free * w1 / total_w)
            m1 = max(1, min(station["modules"] - 1, m1))  # Clamp [1, M-1]
            m2 = station["modules"] - m1
            
            Pshare1 = (w1 / total_w) * station["P_total_kW"]
            Pshare2 = station["P_total_kW"] - Pshare1
    
    elif scenario == "energy_needed":
        if both_active:
            R1 = (target1 - soc1) * ev1["capacity_kWh"]
            R2 = (target2 - soc2) * ev2["capacity_kWh"]
            
            if R1 >= R2:
                m1 += m_free
            else:
                m2 += m_free
        # No Pshare limit
    
    elif scenario == "fifo":
        if both_active:
            # EV1 arrives at t=0, so it gets preference
            m1 += m_free
        # No Pshare limit
    
    elif scenario == "deadline_energy":
        if both_active:
            eps = 0.1
            u1 = 1.0 / max(eps, deadline1 - t)
            u2 = 1.0 / max(eps, deadline2 - t)
            total_u = u1 + u2
            
            m1 += round(m_free * u1 / total_u)
            m1 = max(1, min(station["modules"] - 1, m1))
            m2 = station["modules"] - m1
        # No Pshare limit
    
    elif scenario == "fifo_energy":
        if both_active:
            R1 = (target1 - soc1) * ev1["capacity_kWh"]
            R2 = (target2 - soc2) * ev2["capacity_kWh"]
            
            # If EV2 needs much less, give it extra modules
            if R2 <= 0.25 * R1:
                extra = max(1, int(np.ceil(station["modules"] / 3)))
                m2 += min(extra, m_free)
                m2 = min(m2, station["modules"] - 1)
                m1 = station["modules"] - m2
            else:
                # FIFO: EV1 gets preference
                m1 += m_free
        # No Pshare limit
    
    elif scenario == "deadline":
        # MILP: Deadline-based optimization
        if both_active:
            try:
                prob = LpProblem("Deadline_Allocation", LpMaximize)
                
                I1_var = LpVariable("I_EV1", 0, station["Imax"])
                I2_var = LpVariable("I_EV2", 0, station["Imax"])
                M1_var = LpVariable("M_EV1", 0, station["modules"], cat="Integer")
                M2_var = LpVariable("M_EV2", 0, station["modules"], cat="Integer")
                
                # Constraints
                prob += M1_var + M2_var <= station["modules"], "Total_Modules"
                prob += M1_var >= 1, "Min_M1"
                prob += M2_var >= 1, "Min_M2"
                
                # Module capacity constraints
                prob += I1_var * V1 <= M1_var * station["module_kW"] * 1000, "Module_Cap_1"
                prob += I2_var * V2 <= M2_var * station["module_kW"] * 1000, "Module_Cap_2"
                
                # Curve constraints
                prob += I1_var * V1 <= Pcurve1 * 1000, "Curve_Cap_1"
                prob += I2_var * V2 <= Pcurve2 * 1000, "Curve_Cap_2"
                
                # Current constraints
                prob += I1_var <= station["Imax"], "Current_Cap_1"
                prob += I2_var <= station["Imax"], "Current_Cap_2"
                
                # Total power constraint
                prob += (I1_var * V1 + I2_var * V2) <= station["P_total_kW"] * 1000, "Total_Power"
                
                # Objective: weighted energy delivery
                eps = 0.1
                time_left1 = max(eps, deadline1 - t)
                time_left2 = max(eps, deadline2 - t)
                w1 = 1.0 / time_left1
                w2 = 1.0 / time_left2
                
                E1 = (I1_var * V1 * (1/60)) / 1000  # kWh this step
                E2 = (I2_var * V2 * (1/60)) / 1000
                
                prob += w1 * E1 + w2 * E2, "Objective"
                
                # Solve
                solver = get_solver()
                prob.solve(solver)
                
                if prob.status == 1:  # Optimal
                    I1 = lp_value(I1_var)
                    I2 = lp_value(I2_var)
                    m1 = int(lp_value(M1_var))
                    m2 = int(lp_value(M2_var))
                    
                    # Calculate power for phase classification
                    P1 = (I1 * V1) / 1000
                    P2 = (I2 * V2) / 1000
                    
                    Pmod1 = p_cap_by_modules(m1, station)
                    Pmod2 = p_cap_by_modules(m2, station)
                    
                    phase1, _ = classify_phase(P1, I1, V1, Pcurve1, station["Imax"], Pmod1, float('inf'))
                    phase2, _ = classify_phase(P2, I2, V2, Pcurve2, station["Imax"], Pmod2, float('inf'))
                    
                    return I1, I2, m1, m2, phase1, phase2
                else:
                    raise Exception("MILP not optimal")
                    
            except Exception as e:
                print(f"  ⚠️  MILP failed: {e}, using equal_share fallback")
                # Fallback to equal_share
                m1 = 1 + (m_free // 2)
                m2 = station["modules"] - m1
                Pshare1 = station["P_total_kW"] / 2
                Pshare2 = station["P_total_kW"] / 2
    
    elif scenario == "max_min_fairness":
        # MILP: Max-min fairness optimization
        if both_active:
            try:
                prob = LpProblem("MaxMin_Fairness", LpMaximize)
                
                I1_var = LpVariable("I_EV1", 0, station["Imax"])
                I2_var = LpVariable("I_EV2", 0, station["Imax"])
                M1_var = LpVariable("M_EV1", 0, station["modules"], cat="Integer")
                M2_var = LpVariable("M_EV2", 0, station["modules"], cat="Integer")
                tau = LpVariable("tau", 0)
                
                # Constraints
                prob += M1_var + M2_var <= station["modules"], "Total_Modules"
                prob += M1_var >= 1, "Min_M1"
                prob += M2_var >= 1, "Min_M2"
                
                # Module capacity constraints
                prob += I1_var * V1 <= M1_var * station["module_kW"] * 1000, "Module_Cap_1"
                prob += I2_var * V2 <= M2_var * station["module_kW"] * 1000, "Module_Cap_2"
                
                # Curve constraints
                prob += I1_var * V1 <= Pcurve1 * 1000, "Curve_Cap_1"
                prob += I2_var * V2 <= Pcurve2 * 1000, "Curve_Cap_2"
                
                # Current constraints
                prob += I1_var <= station["Imax"], "Current_Cap_1"
                prob += I2_var <= station["Imax"], "Current_Cap_2"
                
                # Total power constraint
                prob += (I1_var * V1 + I2_var * V2) <= station["P_total_kW"] * 1000, "Total_Power"
                
                # Fairness: each EV's power >= tau
                prob += I1_var * V1 >= tau, "Fair_1"
                prob += I2_var * V2 >= tau, "Fair_2"
                
                # Objective: maximize minimum power
                prob += tau, "Objective"
                
                # Solve
                solver = get_solver()
                prob.solve(solver)
                
                if prob.status == 1:  # Optimal
                    I1 = lp_value(I1_var)
                    I2 = lp_value(I2_var)
                    m1 = int(lp_value(M1_var))
                    m2 = int(lp_value(M2_var))
                    
                    # Calculate power for phase classification
                    P1 = (I1 * V1) / 1000
                    P2 = (I2 * V2) / 1000
                    
                    Pmod1 = p_cap_by_modules(m1, station)
                    Pmod2 = p_cap_by_modules(m2, station)
                    
                    phase1, _ = classify_phase(P1, I1, V1, Pcurve1, station["Imax"], Pmod1, float('inf'))
                    phase2, _ = classify_phase(P2, I2, V2, Pcurve2, station["Imax"], Pmod2, float('inf'))
                    
                    return I1, I2, m1, m2, phase1, phase2
                else:
                    raise Exception("MILP not optimal")
                    
            except Exception as e:
                print(f"  ⚠️  MILP failed: {e}, using equal_share fallback")
                # Fallback to equal_share
                m1 = 1 + (m_free // 2)
                m2 = station["modules"] - m1
                Pshare1 = station["P_total_kW"] / 2
                Pshare2 = station["P_total_kW"] / 2
    
    # ====================
    # STEP 4-8: HEURISTIC POWER & CURRENT CALCULATION (for non-MILP scenarios)
    # ====================
    
    # Module power capacity
    Pmod1 = p_cap_by_modules(m1, station)
    Pmod2 = p_cap_by_modules(m2, station)
    
    # Raw power before total constraint
    P1_raw = min(Pcurve1, Pmod1, (station["Imax"] * V1) / 1000, Pshare1) if ev1_active else 0.0
    P2_raw = min(Pcurve2, Pmod2, (station["Imax"] * V2) / 1000, Pshare2) if ev2_active else 0.0
    
    # Enforce total power constraint
    P1, P2 = enforce_total_power(P1_raw, P2_raw, station)
    
    # Calculate currents
    I1 = (P1 * 1000) / V1 if (ev1_active and V1 > 0) else 0.0
    I2 = (P2 * 1000) / V2 if (ev2_active and not np.isnan(V2) and V2 > 0) else 0.0
    
    # Phase classification
    if ev1_active:
        phase1, _ = classify_phase(P1, I1, V1, Pcurve1, station["Imax"], Pmod1, Pshare1)
    else:
        phase1 = "Done"
    
    if ev2_active:
        phase2, _ = classify_phase(P2, I2, V2, Pcurve2, station["Imax"], Pmod2, Pshare2)
    else:
        phase2 = "Done"
    
    return I1, I2, m1, m2, phase1, phase2

# ============================================================================
# 5. MAIN SIMULATION
# ============================================================================

def run_simulation(ev1, ev2, station, soc1_init, soc2_init, target1, target2,
                   arrival2, deadline1, deadline2, scenario):
    """Run the charging simulation with standardized flow"""

    print("\n" + "="*70)
    print("STARTING SIMULATION")
    print("="*70)
    print(f"Scenario: {scenario.upper()}")
    print(f"EV1: {ev1['capacity_kWh']} kWh ({ev1['chem']}) | SOC: {soc1_init*100:.0f}% → {target1*100:.0f}%")
    print(f"EV2: {ev2['capacity_kWh']} kWh ({ev2['chem']}) | SOC: {soc2_init*100:.0f}% → {target2*100:.0f}%")
    print(f"Station: {station['P_total_kW']} kW, {station['modules']} modules × {station['module_kW']} kW")
    print(f"EV2 arrives at t={arrival2} min")
    print(f"Deadlines: EV1={deadline1} min, EV2={deadline2} min")
    print("="*70)

    # Initialize
    soc1, soc2 = soc1_init, soc2_init
    t = 0
    max_time = max(deadline1, deadline2, 300)

    # Logging arrays - use NaN for not-yet-arrived EVs
    time_log = [0]
    soc1_log = [soc1]
    soc2_log = [soc2]
    v1_log = [pack_voltage(ev1, soc1)]
    v2_log = [np.nan]  # EV2 not yet arrived at t=0
    i1_log = [0]
    i2_log = [np.nan]  # EV2 not yet arrived
    p1_log = [0]
    p2_log = [np.nan]  # EV2 not yet arrived
    m1_log = [0]
    m2_log = [0]
    phase1_log = ["Init"]
    phase2_log = ["Not Arrived"]

    energy1_delivered = 0
    energy2_delivered = 0

    print("\nTime-step simulation output:")
    print("-"*70)
    print(f"{'t':>4} | {'EV1 SOC':>8} {'I1':>7} {'V1':>7} {'P1':>7} {'Phase':>5} | "
          f"{'EV2 SOC':>8} {'I2':>7} {'V2':>7} {'P2':>7} {'Phase':>5}")
    print("-"*70)

    # Simulation loop
    while t < max_time:
        # Check if both EVs are done
        if soc1 >= target1 and (t < arrival2 or soc2 >= target2):
            break

        # Allocate power using standardized flow
        I1, I2, m1, m2, phase1, phase2 = allocate_power(
            ev1, ev2, station, soc1, soc2, t, arrival2, target1, target2,
            scenario, deadline1, deadline2
        )

        # Calculate voltages for logging
        v1 = pack_voltage(ev1, soc1)
        v2 = pack_voltage(ev2, soc2) if t >= arrival2 else np.nan

        # Calculate power
        p1 = (I1 * v1) / 1000 if not np.isnan(v1) else 0
        p2 = (I2 * v2) / 1000 if (not np.isnan(v2) and v2 > 0) else 0

        # Update SOCs
        if soc1 < target1:
            soc1, e1 = energy_update(soc1, I1, v1, ev1["capacity_kWh"])
            energy1_delivered += e1

        if t >= arrival2 and soc2 < target2:
            soc2, e2 = energy_update(soc2, I2, v2, ev2["capacity_kWh"])
            energy2_delivered += e2

        # Increment time before logging
        t += 1

        # Log data with NaN for EV2 before arrival
        time_log.append(t)
        soc1_log.append(soc1)
        soc2_log.append(soc2)
        v1_log.append(v1)
        v2_log.append(v2 if t >= arrival2 else np.nan)
        i1_log.append(I1)
        i2_log.append(I2 if t >= arrival2 else np.nan)
        p1_log.append(p1)
        p2_log.append(p2 if t >= arrival2 else np.nan)
        m1_log.append(m1)
        m2_log.append(m2)
        phase1_log.append(phase1)
        phase2_log.append(phase2 if t >= arrival2 else "Not Arrived")

        # Print every 5 minutes or important events
        if t % 5 == 0 or t == arrival2 or soc1 >= target1 or (t >= arrival2 and soc2 >= target2):
            v2_display = v2 if t >= arrival2 else 0.0
            i2_display = I2 if t >= arrival2 else 0.0
            p2_display = p2 if t >= arrival2 else 0.0
            phase2_display = phase2 if t >= arrival2 else "Wait"

            print(f"{t:4d} | {soc1*100:7.1f}% {I1:6.1f}A {v1:6.1f}V {p1:6.1f}kW {phase1:>5} | "
                  f"{soc2*100:7.1f}% {i2_display:6.1f}A {v2_display:6.1f}V {p2_display:6.1f}kW {phase2_display:>5}")

    print("-"*70)
    print(f"Simulation completed at t={t} minutes")
    print("="*70)

    # Calculate metrics
    theoretical_energy1 = (target1 - soc1_init) * ev1["capacity_kWh"]
    theoretical_energy2 = (target2 - soc2_init) * ev2["capacity_kWh"]

    efficiency1 = (energy1_delivered / theoretical_energy1 * 100) if theoretical_energy1 > 0 else 0
    efficiency2 = (energy2_delivered / theoretical_energy2 * 100) if theoretical_energy2 > 0 else 0

    # Find completion times
    completion1 = next((i for i, s in enumerate(soc1_log) if s >= target1), t)
    completion2 = next((i for i, s in enumerate(soc2_log) if s >= target2), t)

    # Average power (excluding zeros and before arrival for EV2)
    avg_p1 = np.nanmean([p for p in p1_log[1:] if p > 0]) if any(p > 0 for p in p1_log[1:]) else 0
    p2_active = [p2_log[i] for i in range(1, len(p2_log)) if time_log[i] >= arrival2 and not np.isnan(p2_log[i]) and p2_log[i] > 0]
    avg_p2 = np.mean(p2_active) if len(p2_active) > 0 else 0

    results = {
        'time': time_log, 'soc1': soc1_log, 'soc2': soc2_log,
        'v1': v1_log, 'v2': v2_log, 'i1': i1_log, 'i2': i2_log,
        'p1': p1_log, 'p2': p2_log, 'm1': m1_log, 'm2': m2_log,
        'phase1': phase1_log, 'phase2': phase2_log,
        'completion1': completion1, 'completion2': completion2,
        'energy1': energy1_delivered, 'energy2': energy2_delivered,
        'theoretical1': theoretical_energy1, 'theoretical2': theoretical_energy2,
        'efficiency1': efficiency1, 'efficiency2': efficiency2,
        'avg_p1': avg_p1, 'avg_p2': avg_p2,
        'deadline1_met': completion1 <= deadline1,
        'deadline2_met': completion2 <= deadline2,
        'arrival2': arrival2
    }

    return results

# ============================================================================
# 6. PLOTTING FUNCTIONS
# ============================================================================

def plot_results(results, ev1_name, ev2_name):
    """Create all 8 required plots"""

    fig = plt.figure(figsize=(20, 12))

    # Convert SOC to percentage for plots
    soc1_pct = np.array(results['soc1']) * 100
    soc2_pct = np.array(results['soc2']) * 100

    # Get arrival time for markers
    arrival2 = results.get('arrival2', 0)

    # 1. Voltage vs SOC%
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(soc1_pct, results['v1'], 'b-', linewidth=2, label='EV1')
    ax1.plot(soc2_pct, results['v2'], 'r-', linewidth=2, label='EV2')
    ax1.set_xlabel('SOC (%)', fontsize=11)
    ax1.set_ylabel('Voltage (V)', fontsize=11)
    ax1.set_title('Voltage vs SOC', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Current vs SOC%
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(soc1_pct, results['i1'], 'b-', linewidth=2, label='EV1')
    ax2.plot(soc2_pct, results['i2'], 'r-', linewidth=2, label='EV2')
    ax2.set_xlabel('SOC (%)', fontsize=11)
    ax2.set_ylabel('Current (A)', fontsize=11)
    ax2.set_title('Current vs SOC', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Modules vs Time
    ax3 = plt.subplot(3, 3, 3)
    ax3.step(results['time'], results['m1'], 'b-', linewidth=2, label='EV1', where='post')
    ax3.step(results['time'], results['m2'], 'r-', linewidth=2, label='EV2', where='post')
    if arrival2 > 0:
        ax3.axvline(x=arrival2, color='gray', linestyle='--', alpha=0.5, label=f'EV2 arrives')
    ax3.set_xlabel('Time (min)', fontsize=11)
    ax3.set_ylabel('Modules (integer)', fontsize=11)
    ax3.set_title('Module Allocation vs Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(-0.5, max(max(results['m1']), max(results['m2'])) + 0.5)

    # 4. Voltage vs Time
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(results['time'], results['v1'], 'b-', linewidth=2, label='EV1')
    ax4.plot(results['time'], results['v2'], 'r-', linewidth=2, label='EV2')
    if arrival2 > 0:
        ax4.axvline(x=arrival2, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (min)', fontsize=11)
    ax4.set_ylabel('Voltage (V)', fontsize=11)
    ax4.set_title('Voltage vs Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Current vs Time
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(results['time'], results['i1'], 'b-', linewidth=2, label='EV1')
    ax5.plot(results['time'], results['i2'], 'r-', linewidth=2, label='EV2')
    if arrival2 > 0:
        ax5.axvline(x=arrival2, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (min)', fontsize=11)
    ax5.set_ylabel('Current (A)', fontsize=11)
    ax5.set_title('Current vs Time', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # 6. Power vs Time
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(results['time'], results['p1'], 'b-', linewidth=2, label='EV1')
    ax6.plot(results['time'], results['p2'], 'r-', linewidth=2, label='EV2')
    # Calculate total power (handling NaN values)
    p1_arr = np.array(results['p1'])
    p2_arr = np.array(results['p2'])
    p2_arr_clean = np.where(np.isnan(p2_arr), 0, p2_arr)
    total_p = p1_arr + p2_arr_clean
    ax6.plot(results['time'], total_p, 'k--', linewidth=2, label='Total', alpha=0.7)
    if arrival2 > 0:
        ax6.axvline(x=arrival2, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Time (min)', fontsize=11)
    ax6.set_ylabel('Power (kW)', fontsize=11)
    ax6.set_title('Power vs Time', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # 7. Current vs Voltage
    ax7 = plt.subplot(3, 3, 7)
    # Filter out NaN values for scatter plot
    v1_valid = np.array(results['v1'])
    i1_valid = np.array(results['i1'])
    v2_valid = np.array(results['v2'])
    i2_valid = np.array(results['i2'])

    mask1 = ~np.isnan(v1_valid) & ~np.isnan(i1_valid)
    mask2 = ~np.isnan(v2_valid) & ~np.isnan(i2_valid)

    ax7.plot(v1_valid[mask1], i1_valid[mask1], 'b.', markersize=3, label='EV1', alpha=0.6)
    ax7.plot(v2_valid[mask2], i2_valid[mask2], 'r.', markersize=3, label='EV2', alpha=0.6)
    ax7.set_xlabel('Voltage (V)', fontsize=11)
    ax7.set_ylabel('Current (A)', fontsize=11)
    ax7.set_title('Current vs Voltage', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # 8. Power vs SOC%
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(soc1_pct, results['p1'], 'b-', linewidth=2, label='EV1')
    ax8.plot(soc2_pct, results['p2'], 'r-', linewidth=2, label='EV2')
    ax8.set_xlabel('SOC (%)', fontsize=11)
    ax8.set_ylabel('Power (kW)', fontsize=11)
    ax8.set_title('Power vs SOC', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # 9. SOC vs Time
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(results['time'], soc1_pct, 'b-', linewidth=2, label='EV1')
    ax9.plot(results['time'], soc2_pct, 'r-', linewidth=2, label='EV2')
    ax9.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    if arrival2 > 0:
        ax9.axvline(x=arrival2, color='gray', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Time (min)', fontsize=11)
    ax9.set_ylabel('SOC (%)', fontsize=11)
    ax9.set_title('SOC vs Time', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    plt.suptitle(f'EV Charging Simulation Results\n{ev1_name} & {ev2_name}',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig

# ============================================================================
# 7. INTERACTIVE USER INPUT
# ============================================================================

def interactive_simulation():
    """Run interactive simulation with user inputs"""

    print("\n" + "="*70)
    print(" "*15 + "EV SMART CHARGING SIMULATION")
    print(" "*20 + "(UPDATED VERSION)")
    print("="*70)

    # Select EV1
    print("\n📋 Available EV Models:")
    ev_list = list(EVS.keys())
    for i, name in enumerate(ev_list):
        ev = EVS[name]
        print(f"  [{i}] {name} - {ev['capacity_kWh']} kWh ({ev['chem']})")

    while True:
        try:
            idx1 = int(input("\n🚗 Select EV1 (enter number): "))
            if 0 <= idx1 < len(ev_list):
                ev1_name = ev_list[idx1]
                ev1 = EVS[ev1_name]
                break
            print("❌ Invalid selection. Try again.")
        except:
            print("❌ Please enter a number.")

    # Select EV2
    while True:
        try:
            idx2 = int(input("🚗 Select EV2 (enter number): "))
            if 0 <= idx2 < len(ev_list):
                ev2_name = ev_list[idx2]
                ev2 = EVS[ev2_name]
                break
            print("❌ Invalid selection. Try again.")
        except:
            print("❌ Please enter a number.")

    # Select Station
    print("\n⚡ Available Charging Stations:")
    station_list = list(STATIONS.keys())
    for i, name in enumerate(station_list):
        st = STATIONS[name]
        print(f"  [{i}] {name} - {st['P_total_kW']} kW, {st['modules']} modules")

    while True:
        try:
            sidx = int(input("\n🔌 Select Station (enter number): "))
            if 0 <= sidx < len(station_list):
                station_name = station_list[sidx]
                station = STATIONS[station_name]
                break
            print("❌ Invalid selection. Try again.")
        except:
            print("❌ Please enter a number.")

    # Select Scenario
    print("\n🎯 Available Scenarios:")
    for i, sc in enumerate(SCENARIOS):
        print(f"  [{i}] {sc}")

    while True:
        try:
            scidx = int(input("\n📊 Select Scenario (enter number): "))
            if 0 <= scidx < len(SCENARIOS):
                scenario = SCENARIOS[scidx]
                break
            print("❌ Invalid selection. Try again.")
        except:
            print("❌ Please enter a number.")

    # Get SOC values
    print("\n🔋 State of Charge (SOC) Configuration:")
    while True:
        try:
            soc1_init = float(input("  EV1 initial SOC (0-1, e.g., 0.1 for 10%): "))
            if 0 <= soc1_init < 1:
                break
            print("❌ SOC must be between 0 and 1.")
        except:
            print("❌ Please enter a valid number.")

    while True:
        try:
            soc1_target = float(input("  EV1 target SOC (0-1): "))
            if soc1_init < soc1_target <= 1:
                break
            print(f"❌ Target must be between {soc1_init} and 1.")
        except:
            print("❌ Please enter a valid number.")

    while True:
        try:
            soc2_init = float(input("  EV2 initial SOC (0-1): "))
            if 0 <= soc2_init < 1:
                break
            print("❌ SOC must be between 0 and 1.")
        except:
            print("❌ Please enter a valid number.")

    while True:
        try:
            soc2_target = float(input("  EV2 target SOC (0-1): "))
            if soc2_init < soc2_target <= 1:
                break
            print(f"❌ Target must be between {soc2_init} and 1.")
        except:
            print("❌ Please enter a valid number.")

    # Get arrival time
    while True:
        try:
            arrival2 = int(input("\n⏰ EV2 arrival time (minutes from t=0): "))
            if arrival2 >= 0:
                break
            print("❌ Arrival time must be >= 0.")
        except:
            print("❌ Please enter a valid number.")

    # Get deadlines with feasibility checking
    print("\n⏱️  Deadline Configuration:")
    print("(Deadlines will be checked for feasibility...)")

    feasible = False
    while not feasible:
        while True:
            try:
                deadline1 = int(input("  EV1 deadline (minutes from t=0): "))
                if deadline1 > 0:
                    break
                print("❌ Deadline must be > 0.")
            except:
                print("❌ Please enter a valid number.")

        while True:
            try:
                deadline2 = int(input("  EV2 deadline (minutes from t=0): "))
                if deadline2 > arrival2:
                    break
                print(f"❌ Deadline must be > {arrival2} (arrival time).")
            except:
                print("❌ Please enter a valid number.")

        # Check feasibility
        feasible, time1_min, time2_min, messages = check_deadline_feasibility(
            ev1, ev2, station, soc1_init, soc2_init, soc1_target, soc2_target,
            arrival2, deadline1, deadline2, verbose=True
        )

        if not feasible:
            print("\n⚠️  The entered deadlines are not feasible!")
            print("Would you like to:")
            print("  [1] Re-enter deadlines")
            print("  [2] Use recommended deadlines")
            print("  [3] Proceed anyway (simulation may not meet deadlines)")

            choice = input("\nYour choice (1/2/3): ")
            if choice == "2":
                deadline1 = int(np.ceil(time1_min * 1.2))
                deadline2 = arrival2 + int(np.ceil(time2_min * 1.2))
                print(f"\n✅ Using recommended deadlines: EV1={deadline1} min, EV2={deadline2} min")
                feasible = True
            elif choice == "3":
                print("\n⚠️  Proceeding with potentially infeasible deadlines...")
                feasible = True
            # else continue loop for choice 1

    # Run simulation
    results = run_simulation(ev1, ev2, station, soc1_init, soc2_init,
                            soc1_target, soc2_target, arrival2,
                            deadline1, deadline2, scenario)

    # Display results
    print("\n" + "="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    print(f"\n⏱️  Completion Times:")
    print(f"  EV1: {results['completion1']} minutes " +
          ("✅" if results['deadline1_met'] else "❌ MISSED DEADLINE"))
    print(f"  EV2: {results['completion2']} minutes " +
          ("✅" if results['deadline2_met'] else "❌ MISSED DEADLINE"))

    print(f"\n⚡ Energy Delivered:")
    print(f"  EV1: {results['energy1']:.2f} kWh (theoretical: {results['theoretical1']:.2f} kWh)")
    print(f"  EV2: {results['energy2']:.2f} kWh (theoretical: {results['theoretical2']:.2f} kWh)")

    print(f"\n🔋 Charging Efficiency:")
    print(f"  EV1: {results['efficiency1']:.1f}%")
    print(f"  EV2: {results['efficiency2']:.1f}%")

    print(f"\n📊 Average Charging Power:")
    print(f"  EV1: {results['avg_p1']:.1f} kW")
    print(f"  EV2: {results['avg_p2']:.1f} kW")

    print("="*70)

    # Generate plots
    print("\n📈 Generating plots...")
    fig = plot_results(results, ev1_name, ev2_name)
    plt.show()

    print("\n✅ Simulation completed successfully!")

    # Ask if user wants to run another simulation
    again = input("\nRun another simulation? (y/n): ")
    if again.lower() == 'y':
        print("\n" + "="*70)
        interactive_simulation()

# ============================================================================
# 8. MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        interactive_simulation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

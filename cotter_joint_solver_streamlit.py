# cotter_calculator_streamlit.py
# Streamlit cotter-joint calculator (Bhandari-style exact quadratic for d1)
# Run: pip install streamlit numpy
#      streamlit run cotter_calculator_streamlit.py

import streamlit as st
import math
import numpy as np

st.set_page_config(page_title="Cotter Joint Calculator (Bhandari-style)", layout="wide")

st.title("Cotter Joint Calculator — exact quadratic (Bhandari-style)")
st.markdown("Equations use notation: $P$ (N), $d,d_1,d_2$ (mm), $t$ (mm), "
            "$\\sigma_t$ (N/mm$^2$ allowable tensile), $\\tau$ (allowable shear).")

# --- Material databook (representative entries from Bhandari-style table) ---
# NOTE: edit or expand this table to match your edition.
MATERIAL_DB = {
    "30C8 (plain carbon steel)": {"S_ut": 500.0, "S_yt": 400.0, "E": 207000.0},
    "20C8 (plain carbon steel)": {"S_ut": 450.0, "S_yt": 360.0, "E": 207000.0},
    "40C8 (plain carbon steel)": {"S_ut": 600.0, "S_yt": 480.0, "E": 207000.0},
    "Mild steel (generic)": {"S_ut": 350.0, "S_yt": 250.0, "E": 207000.0},
    "Cast iron (FG) (example)": {"S_ut": 200.0, "S_yt": 150.0, "E": 100000.0}
}

st.sidebar.header("Choose material (databook suggestions)")
mat_choice = st.sidebar.selectbox("Material", list(MATERIAL_DB.keys()))
mat_props = MATERIAL_DB[mat_choice]
st.sidebar.write(mat_choice, mat_props)

st.sidebar.header("Inputs")
P_kN = st.sidebar.number_input("Axial tensile load on each rod P (kN)", value=50.0, step=1.0, min_value=0.01)
P = P_kN * 1e3  # N
fos_rods = st.sidebar.number_input("Factor of safety (rod/spigot/socket)", value=6.0, step=0.5, min_value=1.0)
fos_cotter = st.sidebar.number_input("Factor of safety (cotter)", value=4.0, step=0.5, min_value=1.0)
S_yt_material = st.sidebar.number_input("Material yield strength S_yt (N/mm^2) — editable", value=mat_props["S_yt"])
S_yc_factor = st.sidebar.number_input("Assume S_yc = factor * S_yt", value=2.0, step=0.1, min_value=1.0)

# user choice: allow manual control of shear model
st.sidebar.header("Stress model options")
shear_model = st.sidebar.selectbox("Shear model for allowable tau", ["0.5 * S_yt / FOS", "custom value"])
if shear_model.startswith("0.5"):
    # computed later
    custom_tau = None
else:
    custom_tau = st.sidebar.number_input("Custom allowable shear (N/mm^2)", value=50.0)

# slot model factor (k_slot) to allow slight variations between texts
st.sidebar.header("Socket slot model")
k_slot = st.sidebar.number_input("Slot-area factor k_slot in term k_slot*(d1-d2)*t", value=1.0, step=0.1)

# empirical multipliers (defaults suggested, editable)
st.sidebar.header("Empirical multipliers (editable)")
t_factor = st.sidebar.number_input("cotter thickness t = factor * d", value=0.31, step=0.01)
a_factor = st.sidebar.number_input("a = c = factor * d (side margins)", value=0.75, step=0.05)
b_factor = st.sidebar.number_input("b (cotter width) = factor * d", value=1.6, step=0.05)
round_sizes = st.sidebar.checkbox("Round up sizes to nearest mm", value=True)

# Derived allowable stresses
S_yt = S_yt_material
S_yc = S_yc_factor * S_yt
sigma_t_allow = S_yt / fos_rods
tau_allow = (0.5 * S_yt / fos_rods) if custom_tau is None else custom_tau
sigma_c_allow = S_yc / fos_rods

# Show permissibles
st.sidebar.markdown("**Allowable stresses used:**")
st.sidebar.write(f"σ_t (allowable tensile) = {sigma_t_allow:.3f} N/mm^2")
st.sidebar.write(f"τ (allowable shear) = {tau_allow:.3f} N/mm^2")
st.sidebar.write(f"σ_c (allowable crushing) = {sigma_c_allow:.3f} N/mm^2")

# ========== Solver implementation (exact quadratic) ==========
def solve_cotter(P, sigma_t_allow, t_factor=0.31, a_factor=0.75, b_factor=1.6, k_slot=1.0, round_up=True):
    """
    Returns dict with detailed step-by-step numeric values.
    - sigma_t_allow: allowable tensile stress used (N/mm^2)
    - k_slot: coefficient multiplying (d1-d2)*t in socket eqn
    """
    # Step 1: d (gross) from tensile
    d_raw = math.sqrt((4.0 * P) / (math.pi * sigma_t_allow))
    d = math.ceil(d_raw) if round_up else d_raw

    # Step 2: t (cotter thickness)
    t_raw = t_factor * d
    t = math.ceil(t_raw) if round_up else t_raw

    # Step 3: d2 from net-section exact formula
    term = (4.0 * P) / (math.pi * sigma_t_allow)
    if d*d - term < 0:
        d2_raw = float('nan')
        d2 = None
    else:
        d2_raw = math.sqrt(max(0.0, d*d - term))
        d2 = math.ceil(d2_raw) if round_up else d2_raw

    # Step 4: quadratic for d1 from:
    # (pi/4)*sigma_t * (d1^2 - d2^2 - k_slot*(d1 - d2)*t) = P
    A = (math.pi / 4.0) * sigma_t_allow
    B = - (math.pi / 4.0) * sigma_t_allow * (k_slot * t)
    C = (math.pi / 4.0) * sigma_t_allow * (k_slot * t * d2 - d2 * d2) - P

    disc = B*B - 4.0*A*C
    if disc < 0 or not np.isfinite(d2_raw):
        d1_raw = float('nan')
        d1 = None
        roots = (None, None)
    else:
        r1 = (-B + math.sqrt(disc)) / (2.0 * A)
        r2 = (-B - math.sqrt(disc)) / (2.0 * A)
        d1_raw = max(r1, r2)
        d1 = math.ceil(d1_raw) if round_up else d1_raw
        roots = (r1, r2)

    # checks
    a = a_factor * d
    b = b_factor * d
    sc_spigot = P / (t * d2) if (d2 and t) else float('nan')
    tau_spigot = P / (2.0 * a * d2) if (d2 and a) else float('nan')
    sc_socket = P / (t * (d1 - d2)) if (d1 and d2 and t and (d1 - d2) > 0) else float('nan')
    tau_socket = P / (2.0 * d * a)  # approximation

    return {
        'P': P, 'd_raw': d_raw, 'd': d, 't_raw': t_raw, 't': t,
        'd2_raw': d2_raw, 'd2': d2, 'd1_raw': d1_raw, 'd1': d1, 'roots': roots,
        'a': a, 'b': b,
        'checks': {
            'spigot_crushing': sc_spigot,
            'spigot_shear': tau_spigot,
            'socket_crushing': sc_socket,
            'socket_shear': tau_socket,
            'sigma_t_allow': sigma_t_allow
        },
        'a_factor': a_factor, 't_factor': t_factor, 'k_slot': k_slot
    }

# Run solver on current inputs
out = solve_cotter(P=P, sigma_t_allow=sigma_t_allow, t_factor=t_factor, a_factor=a_factor, b_factor=b_factor, k_slot=k_slot, round_up=round_sizes)

# ========== Present results: step-by-step with LaTeX ==========
st.header("Step-by-step derivation & numerical substitution")

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Inputs & permissibles")
    st.write(f"P = {P_kN:.3f} kN = {P:.0f} N")
    st.write(f"Material S_yt (editable) = {S_yt:.3f} N/mm^2")
    st.write(f"Assumed S_yc = {S_yc_factor} * S_yt = {S_yc:.3f} N/mm^2")
    st.write(f"FOS rod/spigot/socket = {fos_rods}, FOS cotter = {fos_cotter}")
    st.latex(r"\sigma_t^{(allow)} = \frac{S_{yt}}{FOS_{\text{rod}}}")
    st.write(f"-> σ_t (allowable tensile) = {sigma_t_allow:.4f} N/mm^2")
    st.latex(r"\tau^{(allow)} = \text{(user model: see sidebar)}")
    st.write(f"-> τ (allowable shear) = {tau_allow:.4f} N/mm^2")

with col2:
    st.subheader("Equation 1 — rod diameter (gross) d")
    st.latex(r"d=\sqrt{\dfrac{4P}{\pi\,\sigma_t^{(allow)}}}")
    st.write(f"Substitute: 4P = {4*P:.0f}, π σ_t = {math.pi*sigma_t_allow:.6f}")
    st.write(f"d (raw) = {out['d_raw']:.6f} mm")
    st.write(f"Selected d = {out['d']} mm")

st.markdown("---")
st.subheader("Spigot net diameter at slot (d2)")
st.latex(r"\frac{\pi}{4}(d^2 - d_2^2)\,\sigma_t^{(allow)} = P \quad\Rightarrow\quad d_2=\sqrt{d^2 - \frac{4P}{\pi\sigma_t^{(allow)}}}")
st.write(f"d2 (raw) = {out['d2_raw']:.6f} mm  -> chosen d2 = {out['d2']} mm")

st.markdown("---")
st.subheader("Socket outside diameter (d1) — quadratic (Bhandari-style)")
st.latex(r"\frac{\pi}{4}\sigma_t^{(allow)}\Bigl(d_1^2 - d_2^2 - k_{slot}(d_1-d_2)t\Bigr) = P")
st.latex(r"\text{where }k_{slot} = " + f"{k_slot}")
st.write("Rearrange to: A d1^2 + B d1 + C = 0 with")
st.write(f"A = (π/4)σ_t = {(math.pi/4.0)*sigma_t_allow:.6f}")
st.write(f"B = - (π/4)σ_t * (k_slot * t) = {- (math.pi/4.0)*sigma_t_allow*(k_slot*out['t']):.6f}")
st.write(f"C = (π/4)σ_t*(k_slot*t*d2 - d2^2) - P = {(math.pi/4.0)*sigma_t_allow*(k_slot*out['t']*out['d2'] - out['d2']**2) - P:.6f}")
st.write(f"Quadratic roots: {out['roots']}")
st.write(f"Chosen d1 (rounded) = {out['d1']} mm (raw {out['d1_raw']:.6f})")

st.markdown("---")
st.subheader("Checks (crushing & shear)")
checks = out['checks']
st.write(f"Spigot crushing σ_c,spigot = P / (t * d2) = {checks['spigot_crushing']:.3f} N/mm^2 (allow {sigma_c_allow:.3f})")
st.write(f"Spigot shear τ_spigot = P / (2 * a * d2) with a = {a_factor}d = {out['a']:.3f} mm -> {checks['spigot_shear']:.3f} N/mm^2 (allow {tau_allow:.3f})")
st.write(f"Socket crushing σ_c,socket = P / (t * (d1-d2)) = {checks['socket_crushing']:.3f} N/mm^2 (allow {sigma_c_allow:.3f})")
st.write(f"Socket shear (approx) = {checks['socket_shear']:.3f} N/mm^2 (allow {tau_allow:.3f})")

st.markdown("---")
st.subheader("Final recommended sizes (for drawing / manufacture)")
st.write({
    'Rod dia d (mm)': out['d'],
    'Cotter thickness t (mm)': out['t'],
    'Cotter width b (mm)': round(out['b']),
    'Spigot d2 (mm)': out['d2'],
    'Socket d1 (mm)': out['d1'],
    'Side margins a=c (mm)': round(out['a'])
})

st.markdown("---")
st.info("Notes: You can change the 'k_slot' factor to match different textbook variants of the slot-area correction. Material table is editable in SOURCE; cite Bhandari's databook for official tables.")

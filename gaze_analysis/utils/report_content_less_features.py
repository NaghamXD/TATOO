"""
Plain-language content for per-patient OT reports (script 18).

Holds the "translation dictionary" that turns raw feature names into
clinically-readable labels, domain groupings, and short explanatory notes —
kept separate from the report-generation plumbing so wording can be tuned
without touching data-loading/rendering code.
"""

# ── Game descriptions (shown at the top of each per-game section) ───────────
GAME_DESCRIPTIONS = {
    "TouchIt": (
        "A simple tap-the-target task. Measures basic visuomotor response: "
        "how quickly and accurately the patient can see a target and touch it."
    ),
    "CornerIt": (
        "Targets appear in alternating screen corners, requiring the patient "
        "to repeatedly shift attention and reach across the screen. Adds a "
        "spatial-attention and reach-planning demand on top of basic tapping."
    ),
    "DoubleTapIt": (
        "Requires two rapid, precisely-placed taps on the same target. "
        "Probes fine motor timing, response control, and the ability to "
        "repeat a precise action quickly."
    ),
    "PinchIt": (
        "A two-finger pinch (resize) gesture task. Probes bimanual fine-motor "
        "coordination, grip-force control, and smooth multi-finger movement."
    ),
    "SlideIt": (
        "Requires sliding an object along a guided path. Probes sustained, "
        "controlled dragging movements and the ability to keep a steady "
        "trajectory over time."
    ),
    "DragIt": (
        "Requires picking up an object and placing it at a target location. "
        "Probes movement planning, trajectory control, and precision "
        "placement — a more complex, multi-step motor action."
    ),
}

# ── Clinical domains (used to group features within a game section) ─────────
DOMAINS = {
    "motor_planning": {
        "label": "Motor Planning & Execution",
        "blurb": (
            "How the patient prepares for a movement versus how they "
            "physically carry it out — reflects processing speed and the "
            "balance between planning and doing."
        ),
    },
    "motor_control": {
        "label": "Motor Control & Steadiness",
        "blurb": (
            "The smoothness and consistency of the patient's touch — "
            "reflects fine-motor steadiness, grip-force regulation, and "
            "presence of tremor."
        ),
    },
    "precision": {
        "label": "Spatial Precision & Impulsivity",
        "blurb": (
            "How accurately the patient targets the intended location, and "
            "how often they act before/without precise targeting — reflects "
            "spatial precision and impulse control."
        ),
    },
    "visual_attention": {
        "label": "Visual Attention",
        "blurb": (
            "How much of the patient's gaze stayed on the task versus "
            "wandered off-screen, and how their visual fixations were "
            "distributed — reflects sustained, on-task attention."
        ),
    },
    "visual_search": {
        "label": "Visual Search & Scanning",
        "blurb": (
            "The speed and span of the patient's rapid eye movements between "
            "fixations — reflects how efficiently they visually scan and "
            "search the screen."
        ),
    },
    "coordination": {
        "label": "Eye-Hand Coordination",
        "blurb": (
            "How closely the patient's gaze and hand movements were aligned "
            "in space, direction, and timing — a marker of cognitive-motor "
            "integration (i.e. how well 'looking' and 'doing' work together)."
        ),
    },
    "engagement": {
        "label": "Engagement & Persistence",
        "blurb": (
            "How long the patient stayed engaged with the task — can reflect "
            "task endurance, motivation, or fatigue."
        ),
    },
}

# ── Feature → domain/label/unit/note mapping ────────────────────────────────
# `note` explains what the metric reflects, in plain language, without making
# a "good/bad" clinical judgment call — that's left to the OT, informed by the
# peer-comparison band that the report computes alongside it.
FEATURES = {
    "First_Reaction_Time_sec": {
        "label": "First Reaction Time",
        "unit": "sec",
        "domain": "motor_planning",
        "note": "How quickly the patient noticed and began responding.",
    },
    "Flight_Touch_Ratio": {
        "label": "Flight-to-Touch Ratio",
        "unit": "",
        "domain": "motor_planning",
        "note": "Time planning/traveling vs. time physically executing the touch.",
    },
    "Mean_Action_Duration_sec": {
        "label": "Average Action Duration",
        "unit": "sec",
        "domain": "motor_planning",
        "note": "Average time to complete one full pinch, slide, or drag.",
    },
    "Spatial_Jerk_Magnitude": {
        "label": "Movement Jerk (smoothness)",
        "unit": "cm/s³",
        "domain": "motor_control",
        "note": "How smooth vs. jerky the dragging motion was.",
    },
    "Pressure_High_Pct": {
        "label": "% Time at High Touch Pressure",
        "unit": "%",
        "domain": "motor_control",
        "note": "Share of the task spent pressing with high force.",
    },
    "Pressure_Low_Pct": {
        "label": "% Time at Low Touch Pressure",
        "unit": "%",
        "domain": "motor_control",
        "note": "Share of the task spent pressing with light force.",
    },
    "Mean_Pressure_Overall": {
        "label": "Average Touch Pressure",
        "unit": "",
        "domain": "motor_control",
        "note": "Average force applied to the screen across the task.",
    },
    "Pressure_Jitter_Stationary": {
        "label": "Pressure Jitter While Holding Still",
        "unit": "",
        "domain": "motor_control",
        "note": "Pressure fluctuation while still — a marker of micro-tremor.",
    },
    "Number_of_Taps": {
        "label": "Number of Taps",
        "unit": "",
        "domain": "precision",
        "note": "Total number of taps made during the task.",
    },
    "Correct_Taps": {
        "label": "Correct Taps",
        "unit": "",
        "domain": "precision",
        "note": "Taps that landed on the intended target.",
    },
    "Outside_Taps": {
        "label": "Taps Outside Target",
        "unit": "",
        "domain": "precision",
        "note": "Taps that missed the intended target area.",
    },
    "Accuracy_Ratio": {
        "label": "Accuracy Ratio",
        "unit": "",
        "domain": "precision",
        "note": "Correct taps vs. taps outside target — precision & impulsivity.",
    },
    "Gaze_In_Out_Ratio": {
        "label": "Gaze In/Out Ratio",
        "unit": "",
        "domain": "visual_attention",
        "note": "Normalized measure of sustained, on-task visual attention.",
    },
    "Fixation_Count": {
        "label": "Number of Fixations",
        "unit": "",
        "domain": "visual_attention",
        "note": "Distinct moments the gaze settled and held steady.",
    },
    "Mean_Fixation_Duration_sec": {
        "label": "Average Fixation Duration",
        "unit": "sec",
        "domain": "visual_attention",
        "note": "How long gaze held steady on one point before moving on.",
    },
    "Mean_Saccadic_Amplitude_cm": {
        "label": "Average Saccade Distance",
        "unit": "cm",
        "domain": "visual_search",
        "note": "Distance covered by each rapid eye movement between fixations.",
    },
    "Peak_Saccadic_Velocity_cm_s": {
        "label": "Peak Saccade Speed",
        "unit": "cm/s",
        "domain": "visual_search",
        "note": "Average top speed reached during rapid eye movements.",
    },
    "Saccade_Frequency_Hz": {
        "label": "Saccade Frequency",
        "unit": "Hz",
        "domain": "visual_search",
        "note": "How often the eyes made rapid scanning movements.",
    },
    "Mean_Eye_Hand_Distance_cm": {
        "label": "Average Eye-Hand Distance",
        "unit": "cm",
        "domain": "coordination",
        "note": "Spatial distance between gaze point and touch point.",
    },
    "Directional_Velocity_Alignment": {
        "label": "Eye-Hand Direction Alignment",
        "unit": "",
        "domain": "coordination",
        "note": "How closely eye and hand movement directions matched.",
    },
    "True_Eye_Hand_Latency_sec": {
        "label": "Eye-to-Hand Latency",
        "unit": "sec",
        "domain": "coordination",
        "note": "Delay between looking somewhere and the hand moving there.",
    },
    "Game_Duration_sec": {
        "label": "Total Game Duration",
        "unit": "sec",
        "domain": "engagement",
        "note": "Total time actively engaged with this game.",
    },
}

# ── Domain display order within a game section ──────────────────────────────
DOMAIN_ORDER = [
    "motor_planning",
    "motor_control",
    "precision",
    "visual_attention",
    "visual_search",
    "coordination",
    "engagement",
]

MODALITY_LABELS = {
    "Touch": "motor/touch behaviors",
    "Gaze": "visual-attention behaviors",
    "Sync": "eye-hand coordination behaviors",
}


def percentile_band(pct):
    """Translate a 0-100 percentile into a plain-language peer-comparison phrase."""
    if pct is None:
        return "not available for comparison"
    if pct < 25:
        return "lower than most peers of the same age group"
    if pct > 75:
        return "higher than most peers of the same age group"
    return "within the typical range for peers of the same age group"


def modality_summary(touch_pct, gaze_pct, sync_pct, dominant):
    """Plain-language sentence describing the SHAP modality breakdown for a patient."""
    dominant_label = MODALITY_LABELS.get(dominant, dominant)
    return (
        f"Across all games, the data-driven assessment of this patient was "
        f"most strongly characterized by {dominant_label} "
        f"({touch_pct:.0f}% touch / {gaze_pct:.0f}% gaze / {sync_pct:.0f}% "
        f"eye-hand sync contribution to the overall pattern)."
    )

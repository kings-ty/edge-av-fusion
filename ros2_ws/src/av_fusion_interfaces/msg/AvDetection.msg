# Fused audio-visual detection for blind-spot warning.
std_msgs/Header header

# acoustic classification
string sound_class            # vehicle_engine | horn | siren | tire_road | background
float32 confidence            # vehicle probability [0,1]
bool classifier_degraded      # true = energy-trigger fallback active

# direction of arrival (array frame: 0 = broadside, + = left)
float32 doa_deg               # [-90, +90]
float32 doa_confidence        # [0,1], includes endfire geometry penalty
bool front_back_ambiguous     # 2-mic linear array mirror ambiguity unresolved
float32 front_log_odds        # >0: front hypothesis more likely

# fusion
string fusion_state           # idle | candidate | triggered | confirmed
bool vision_confirmed
bool alert                    # downstream consumers should act on this
int32 track_id

# diagnostics
float32 pipeline_latency_ms   # hop read -> message build, this hop

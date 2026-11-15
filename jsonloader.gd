extends AudioStreamPlayer

var data_file_path = "res://songs/I Wanna Rock by Twisted Sisters.analysis.json"
signal _onBeat(beat_data)
signal _onOnset(onset_data)

# ---------------------------------------------------------------------------------------
# Spawn-event tuning (all adjustable from the Inspector)
#
# Instead of firing _onOnset for every onset in the JSON, _ready() builds a spawn schedule:
#   1. the song is split into "calm" and "intense" sections from its loudness + onset density
#   2. calm sections    -> few enemies, on-beat only, enemy_type "spiked"
#   3. intense sections -> more enemies, enemy_type "standard" (no spikes)
# Only events in the schedule are emitted.
# ---------------------------------------------------------------------------------------
@export_group("Section detection")
@export var section_window_sec := 6.0          # smoothing: bigger = fewer, longer sections
@export_range(0.0, 1.0) var intense_percentile := 0.75   # loudest ~32% of the song counts as intense
@export_range(0.0, 0.5) var section_hysteresis := 0.15   # stops flickering around the threshold
@export var min_section_sec := 6.0             # shorter sections are merged into their neighbour

@export_group("Calm sections (spiked)")
@export var calm_beats_per_enemy := 2.0        # at most one enemy per this many beats
@export_range(0.0, 1.0) var calm_min_strength := 0.4
@export var calm_min_gap_sec := 0.6

@export_group("Intense sections (no spikes)")
@export var intense_beats_per_enemy := 1.5     # lower = more enemies (1.0 = about one per beat)
@export_range(0.0, 1.0) var intense_min_strength := 0.25
@export var intense_min_gap_sec := 0.35

@export_group("Timing")
@export var beat_tolerance := 0.12             # how close to a beat (in beats) counts as "on the beat"
@export var stale_event_sec := 0.5             # events later than this (e.g. after a seek) are skipped

var song_data = {}
var beatIndex = 0
var eventIndex = 0
var lifeTime = 0.0
var is_loaded = false

var spawn_schedule: Array = []
var _section_flags: Array = []   # one bool per intensity sample: true = intense
var _section_hz := 20.0


func load_json_file(filePath: String) -> Dictionary:
	if FileAccess.file_exists(filePath):
		var data_file = FileAccess.open(filePath, FileAccess.READ)
		var parsed_data_file = JSON.parse_string(data_file.get_as_text())

		if parsed_data_file is Dictionary:
			return parsed_data_file
		else:
			print("Error reading or parsing JSON: " + filePath)
	else:
		print("File " + filePath + " does not exist")
	return {}


func _ready() -> void:
	song_data = load_json_file(data_file_path)
	if song_data.has("beats") and song_data.has("onsets"):
		is_loaded = true
		_build_section_flags()
		_build_spawn_schedule()
	else:
		print("Song data is missing 'beats' or 'onsets' keys!")


func get_next_beat_delta() -> float:
	if not is_loaded or beatIndex >= song_data["beats"].size():
		return 0.0
	return song_data["beats"][beatIndex]["time"] - lifeTime


# Call this when the song restarts (retry, loop) so beats and spawn events start over.
func restart_emitters() -> void:
	beatIndex = 0
	eventIndex = 0
	lifeTime = 0.0


func _process(dt: float) -> void:
	if not is_loaded:
		return

	lifeTime = self.get_playback_position()

	var beats = song_data["beats"]
	if beatIndex < beats.size():
		if beats[beatIndex]["time"] <= lifeTime:
			var next_beat = beats[beatIndex + 1] if beatIndex + 1 < beats.size() else null
			_onBeat.emit([beats[beatIndex], next_beat])
			beatIndex += 1

	# Emit only the scheduled spawn events.
	# Payload keeps the old layout: [event, next_event, deadline_onset]
	#   event["enemy_type"] -> "spiked" (calm sections) or "standard" (intense sections)
	#   event["section"]    -> "calm" or "intense"
	var onsets: Array = song_data["onsets"]
	while eventIndex < spawn_schedule.size() and spawn_schedule[eventIndex]["time"] <= lifeTime:
		var ev: Dictionary = spawn_schedule[eventIndex]
		eventIndex += 1
		if lifeTime - float(ev["time"]) > stale_event_sec:
			continue
		var next_event = spawn_schedule[eventIndex] if eventIndex < spawn_schedule.size() else null
		# Same deadline as before: the raw onset 10 onsets after this one.
		var raw_index: int = ev["onset_index"]
		var deadline = onsets[raw_index + 3] if raw_index + 3 < onsets.size() else null
		if deadline == null:
			# Near the end of the song there is no onset 10 ahead; avoid a null crash downstream.
			deadline = {"time": float(ev["time"]) + 2.0, "strength": 0.0}
		_onOnset.emit([ev, next_event, deadline])


# ---------------------------------------------------------------------------------------
# Section detection: "calm" vs "intense", relative to the song itself
# ---------------------------------------------------------------------------------------
func _build_section_flags() -> void:
	_section_flags = []
	var inten: Dictionary = song_data.get("intensity", {})
	if not inten.has("loudness_normalized") or not inten.has("sample_rate_hz"):
		push_warning("No intensity data in the analysis JSON: all enemies use the 'intense' rules.")
		return
	var loud: Array = inten["loudness_normalized"]
	_section_hz = float(inten["sample_rate_hz"])
	var n := loud.size()
	if n == 0:
		return

	# onsets per intensity sample = onset density
	var dens: Array = []
	dens.resize(n)
	dens.fill(0.0)
	for o in song_data["onsets"]:
		var idx := int(float(o["time"]) * _section_hz)
		if idx >= 0 and idx < n:
			dens[idx] += 1.0

	var window := maxi(int(section_window_sec * _section_hz), 1)
	var loud_n := _normalize(_moving_average(loud, window))
	var dens_n := _normalize(_moving_average(dens, window))
	var energy: Array = []
	energy.resize(n)
	for i in n:
		energy[i] = 0.6 * float(loud_n[i]) + 0.4 * float(dens_n[i])

	# threshold relative to this song, with hysteresis
	var hi := _percentile(energy, intense_percentile)
	var lo := hi - section_hysteresis
	var intense := false
	_section_flags.resize(n)
	for i in n:
		if not intense and float(energy[i]) > hi:
			intense = true
		elif intense and float(energy[i]) < lo:
			intense = false
		_section_flags[i] = intense

	_enforce_min_section(_section_flags, int(min_section_sec * _section_hz))


func _mode_at(t: float) -> String:
	if _section_flags.is_empty():
		return "intense"
	var i := clampi(int(t * _section_hz), 0, _section_flags.size() - 1)
	return "intense" if _section_flags[i] else "calm"


func _enforce_min_section(flags: Array, min_len: int) -> void:
	# Flip any run shorter than min_len to the state of its neighbour, until stable.
	var changed := true
	var guard := 0
	while changed and guard < 50:
		changed = false
		guard += 1
		var n := flags.size()
		var start := 0
		while start < n:
			var end := start
			while end < n and flags[end] == flags[start]:
				end += 1
			if end - start < min_len and not (start == 0 and end == n):
				var new_val: bool = flags[start - 1] if start > 0 else flags[end]
				for k in range(start, end):
					flags[k] = new_val
				changed = true
				break
			start = end


# ---------------------------------------------------------------------------------------
# Spawn schedule: pick the best onset per time slot, instead of using every onset
# ---------------------------------------------------------------------------------------
func _build_spawn_schedule() -> void:
	spawn_schedule = []
	var onsets: Array = song_data["onsets"]
	var i := 0
	while i < onsets.size():
		var slot_start := float(onsets[i]["time"])
		var mode := _mode_at(slot_start)
		var slot_end := slot_start + _slot_length(slot_start, mode)

		# the best-scoring onset inside this slot wins; everything else is dropped
		var best := -1
		var best_score := 0.0
		var j := i
		while j < onsets.size() and float(onsets[j]["time"]) < slot_end:
			var s := _score_onset(onsets[j], mode)
			if s > best_score:
				best_score = s
				best = j
			j += 1
		if best >= 0:
			_accept(best, mode, best_score)
		i = maxi(j, i + 1)

	for ev in spawn_schedule:
		ev.erase("_score")

	var calm_count := 0
	for ev in spawn_schedule:
		if ev["section"] == "calm":
			calm_count += 1
	print("Spawn schedule: %d events from %d onsets (%d calm/spiked, %d intense/standard)" % [
		spawn_schedule.size(), onsets.size(), calm_count, spawn_schedule.size() - calm_count])


func _slot_length(t: float, mode: String) -> float:
	var beats_per_slot := calm_beats_per_enemy if mode == "calm" else intense_beats_per_enemy
	return beats_per_slot * 60.0 / _bpm_at(t)


func _min_gap(t: float, mode: String) -> float:
	var floor_gap := calm_min_gap_sec if mode == "calm" else intense_min_gap_sec
	return maxf(floor_gap, 0.5 * _slot_length(t, mode))


func _score_onset(onset: Dictionary, mode: String) -> float:
	var strength := float(onset["strength"])
	var min_strength := calm_min_strength if mode == "calm" else intense_min_strength
	if strength < min_strength:
		return 0.0
	var align := _beat_alignment(onset)
	if mode == "calm" and align < 0.65:
		return 0.0  # slow parts: only on-beat or half-beat hits, so spikes can be timed precisely
	return strength * align


func _beat_alignment(onset: Dictionary) -> float:
	var bp = onset.get("beat_position")  # null when the analysis had no beat grid there
	if bp == null:
		return 0.8  # unknown: don't punish it
	var frac: float = float(bp) - floorf(float(bp))
	if minf(frac, 1.0 - frac) <= beat_tolerance:
		return 1.0   # on the beat
	if absf(frac - 0.5) <= beat_tolerance:
		return 0.65  # half-beat (eighth note)
	return 0.3


func _accept(idx: int, mode: String, score: float) -> void:
	var onset: Dictionary = song_data["onsets"][idx]
	var t := float(onset["time"])
	var gap := _min_gap(t, mode)
	# keep events at least `gap` apart; when two collide the higher score survives
	while not spawn_schedule.is_empty():
		var last: Dictionary = spawn_schedule[spawn_schedule.size() - 1]
		if t - float(last["time"]) >= gap:
			break
		if score > float(last["_score"]):
			spawn_schedule.pop_back()
		else:
			return
	var ev := onset.duplicate()
	ev["section"] = mode
	ev["enemy_type"] = "spiked" if mode == "calm" else "standard"
	ev["onset_index"] = idx
	ev["_score"] = score
	spawn_schedule.append(ev)


# ---------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------
func _bpm_at(t: float) -> float:
	var beats: Array = song_data["beats"]
	if beats.is_empty():
		return 120.0
	var lo := 0
	var hi := beats.size() - 1
	while lo < hi:
		var mid := int((lo + hi + 1) / 2.0)
		if float(beats[mid]["time"]) <= t:
			lo = mid
		else:
			hi = mid - 1
	var v = beats[lo].get("local_bpm")
	if v == null:
		return 120.0
	return clampf(float(v), 40.0, 240.0)


func _moving_average(values: Array, window: int) -> Array:
	var n := values.size()
	var prefix: Array = [0.0]
	for v in values:
		prefix.append(float(prefix[prefix.size() - 1]) + float(v))
	var half := int(window / 2.0)
	var out: Array = []
	out.resize(n)
	for i in n:
		var a := maxi(0, i - half)
		var b := mini(n, i + half + 1)
		out[i] = (float(prefix[b]) - float(prefix[a])) / float(b - a)
	return out


func _percentile(values: Array, p: float) -> float:
	if values.is_empty():
		return 0.0
	var sorted_values := values.duplicate()
	sorted_values.sort()
	return float(sorted_values[clampi(int(p * (sorted_values.size() - 1)), 0, sorted_values.size() - 1)])


func _normalize(values: Array) -> Array:
	# maps the 5th..95th percentile of the song onto 0..1
	var lo := _percentile(values, 0.05)
	var hi := _percentile(values, 0.95)
	var out: Array = []
	out.resize(values.size())
	for i in values.size():
		out[i] = 0.0 if hi - lo < 0.000001 else clampf((float(values[i]) - lo) / (hi - lo), 0.0, 1.0)
	return out

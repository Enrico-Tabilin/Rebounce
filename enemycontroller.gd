extends Node2D

var rng = RandomNumberGenerator.new()
var enemyPositions: Array[Enemy] = []
var enemiesRecieveing: Array[Enemy] = []
const TOPLEFT = Vector2(-240, -215)
const BOTTOMRIGHT = Vector2(215, 240)

const TEXTURE_DICTIONARY = {
	1: "res://Enemies/archer.png",
	2: "res://Enemies/shield.png",
	3: "res://Enemies/shieldShort.png",
	2.1: "res://Enemies/shieldSpike.png",
	3.1: "res://Enemies/shieldShortSpike.png",
}

func _clearEnemies() -> void:
	var list = self.get_children()
	for i in range(list.size()):
		list[i].queue_free()
	pass

func _spawnEnemy(params: Dictionary) -> void:
	var enemy = Enemy.new()
	
	for key in params:
		if key in enemy:
			enemy.set(key, params[key])
	
	add_child(enemy)

func _registerEnemy(enemy: Enemy) -> void:
	enemyPositions.append(enemy)

func _on_music_player__on_onset(info) -> void:
	var current_onset = info[0]
	print(info[0])
	if not current_onset:
		return
	
	rng.seed = int(current_onset["time"] * current_onset["strength"])
	
	var hurtsOnImmune = false
	var keys = TEXTURE_DICTIONARY.keys()
	var random_key = rng.randi_range(1,3)
	var isRushable = true
	var isJumpable = true
	if info[0]["enemy_type"] == "spiked" and random_key != 1:
		random_key += 0.1
		hurtsOnImmune = true	
	var chosen_texture_path = TEXTURE_DICTIONARY[random_key]
	var Ranged = false; var isCollideable = false; var deadlineToAttack = 0.0
	var isBodyguard = true; var isCelebrity = false
	if int(random_key) == 2:
		isCollideable = true
		isRushable = false
	if int(random_key) == 3:
		isCollideable = true
		isJumpable = false
		if info[2] != null:
			deadlineToAttack = info[2]["time"] - current_onset["time"]
	if random_key == 1:
		Ranged = true
		isBodyguard = false; isCelebrity = true
		if info[2] != null:
			deadlineToAttack = info[2]["time"] - current_onset["time"]
	
	if enemiesRecieveing.size() > 1:
		if enemiesRecieveing[1]:
			enemiesRecieveing[1].DeadlineToAttack = info[2]["time"] - current_onset["time"]
			enemiesRecieveing.remove_at(1)
	_spawnEnemy({
		"Speed": current_onset["strength"] * 40,
		"position": Vector2(
			rng.randi_range(TOPLEFT.x, BOTTOMRIGHT.x), 
			rng.randi_range(TOPLEFT.y, BOTTOMRIGHT.y)
		),
		"scale": Vector2(1.75,1.75),
		"Sprite": load(chosen_texture_path),
		"DeadlineToAttack": deadlineToAttack,
		"TextureFilter": TEXTURE_FILTER_NEAREST,
		"Ranged": Ranged,
		"isCollideable": isCollideable,
		"isBodyguard": isBodyguard,
		"isCelebrity": isCelebrity,
		"hurtsOnImmune": hurtsOnImmune,
		"isRushable": isRushable,
		"isJumpable": isJumpable,
	})

func _process(delta: float) -> void:
	var pairs = get_children()
	for i in range(pairs.size()):
		if pairs[i].is_in_group("enemies") and pairs[i] is Enemy:
			if pairs[i].Ranged == true and is_inf(pairs[i].DeadlineToAttack):
				enemiesRecieveing.insert(0, pairs[i])
				pass
	pass

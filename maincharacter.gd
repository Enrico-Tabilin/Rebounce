extends CharacterBody2D

const SPEED = 125.0
const JUMP_VELOCITY = -400.0
const FTS = 270
const CHARGE_HIT_SPEED = SPEED * 1.2
const ATTACK_DAMAGE = 1
const ENEMY_BOUNCE_SPEED = 240.0

@export var z : float = 0
var late : bool = true
var timeTillNextBounce = 1
@export var MAX_HEALTH = 3

var health : int = MAX_HEALTH
var attackArea : Area2D
var isCharging : bool = false
var isJumping : bool = false
var alreadyHit : Array = []

signal health_changed
signal died

func _ready() -> void:
	attackArea = Area2D.new()
	attackArea.collision_layer = 0
	attackArea.collision_mask = 1
	var cs := CollisionShape2D.new()
	var circle := CircleShape2D.new()
	circle.radius = 18
	cs.shape = circle
	attackArea.add_child(cs)
	add_child(attackArea)

const HIT_RADIUS = 18.0

func takeDamage(num : int, immune : Array) -> void:
	if immune.find("Charging") and isCharging:
		return
	health -= num
	health_changed.emit(health, MAX_HEALTH or 3)
	if health <= 0:
		died.emit()
		SceneManager.change_scene("res://menu.tscn")
		health = 3
	pass

func _hurtOverlapping(type : int) -> void:
	for node in get_tree().get_nodes_in_group("enemies"):
		var enemy := node as Enemy
		if enemy == null or alreadyHit.has(enemy):
			continue
		var reach = HIT_RADIUS + 12.0 * enemy.scale.x
		if global_position.distance_to(enemy.global_position) > reach:
			continue

		alreadyHit.append(enemy)
		var immune = (type == 1 and not enemy.isRushable) or (type == 2 and not enemy.isJumpable)
		if immune:
			if enemy.hurtsOnImmune == true:
				takeDamage(1, [])
		else:
			enemy.takeDamage(ATTACK_DAMAGE)

func _bounce_towards_mouse() -> void:
	var dir = get_local_mouse_position().normalized(); var df : Vector2 = dir
	velocity -= df*SPEED*1.5
	
	await get_tree().create_timer(0.3).timeout
	velocity += df*SPEED*3.5
	alreadyHit.clear()
	isCharging = true 
	
func reverb(vector : Vector2) -> void:
	velocity += vector
	pass

func _bounce_upwards() -> void:
	var target_pos = get_global_mouse_position()
	var true_pos = target_pos
	var endVelocity = Vector2.ZERO
	
	var ray = RayCast2D.new()
	ray.position = Vector2.ZERO
	ray.target_position = get_local_mouse_position() 
	ray.collision_mask = 1

	add_child(ray)

	ray.force_raycast_update()
	
	if ray.is_colliding():
		var place = ray.get_collision_point()
		endVelocity = -(target_pos - place) / 35 * SPEED
		true_pos = place + (endVelocity)/SPEED*5
	else:
		true_pos = target_pos
	ray.queue_free()
	
	var percentDone = (abs(target_pos.x / true_pos.x) + abs(target_pos.y / true_pos.y))/2
		
	velocity = Vector2.ZERO
	alreadyHit.clear()
	isJumping = true
	var tw = create_tween()
	tw.parallel().tween_property(self, "z", 10, min(percentDone/2,1))
	tw.parallel().tween_property(self, "position", true_pos, min(percentDone,1)).set_ease(Tween.EASE_OUT)
	tw.parallel().tween_property($Sprite2D, "offset", -Vector2(0,15), min(percentDone/2,0.7))
	tw.parallel().tween_property($Sprite2D2, "scale", Vector2(0.2,0.1), min(percentDone/2,0.5))
	tw.parallel().tween_property($Sprite2D2, "scale", Vector2(0.1,0.05), min(percentDone/2,0.5)).set_delay(0.5)
	tw.parallel().tween_property($Sprite2D, "offset", Vector2.ZERO, min(percentDone/2,0.3)).set_delay(0.7)
	tw.finished.connect(func():
		_hurtOverlapping(2) 
		isJumping = false 
		_pushOutOfSolids()
		)
	
	velocity += endVelocity

func _pushOutOfSolids() -> void:
	if not get_tree():
		return
	for node in get_tree().get_nodes_in_group("enemies"):
		var enemy := node as Enemy
		if enemy == null or not enemy.isCollideable:
			continue
		var min_dist = HIT_RADIUS + 12.0 * enemy.scale.x
		var offset = global_position - enemy.global_position
		if offset.length() < min_dist:
			var away = offset.normalized() if offset.length() > 0.01 else Vector2.RIGHT
			global_position = enemy.global_position + away * min_dist
			
func _physics_process(dt: float) -> void:
	var direction : Vector2 = Vector2.ZERO
	$AnimationPlayer.speed_scale = (FTS/timeTillNextBounce)
	if z <= 0.1:
		direction = Input.get_vector("a","d","w","s")
		if Input.is_action_just_pressed("click"):
			_bounce_towards_mouse()
		if Input.is_action_just_pressed("other"):
			$AnimationPlayer.play("RESET")
			_bounce_upwards()
		else:
			if $AnimationPlayer.is_playing() == false:
				$AnimationPlayer.play("idle")
	else:
		z = max(z-10*dt,0)
	velocity.x = move_toward(velocity.x, 0, SPEED*dt); velocity.y = move_toward(velocity.y, 0, SPEED*dt)
	
	$Sprite2D.scale = Vector2.ONE*(1+z**2/6 + $Sprite2D.offset.y*-1/16)
		
	velocity += direction*SPEED
	var pre_velocity = velocity
	move_and_slide()
	
	if z <= 0.1:
		for i in get_slide_collision_count():
			var col = get_slide_collision(i)
			var enemy = col.get_collider() as Enemy
			if enemy and enemy.isCollideable:
				var n = col.get_normal()
				velocity = pre_velocity.bounce(n) * 0.6 + n * ENEMY_BOUNCE_SPEED * 0.5
				isCharging = false
				break
				
	velocity -= direction*SPEED

	if isCharging and velocity.length() < CHARGE_HIT_SPEED:
		isCharging = false
	if isCharging:
		_hurtOverlapping(1)

func _on_bouncy_entered(body: Node2D) -> void:
	velocity *= -0.85
	position = position*0.95
	pass

func _on_music_player__on_beat(info) -> void:
	if info[1] != null:
		timeTillNextBounce = (info[1]["time"] - info[0]["time"])
		print(info[1]["time"])
	pass

func _on_music_player__on_onset(info) -> void:
	if info[1] != null:
		print(info[1]["time"])
	pass

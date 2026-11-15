class_name Enemy
extends CharacterBody2D

@export var Collision_Shape : CollisionShape2D
@export var Sprite : Texture2D
@export var Speed : float = 120.0
@export var isBodyguard : bool = false
@export var isCelebrity : bool = false
@export var DeadlineToAttack : float = 3.0
@export var isJumpable : bool = false
@export var isRushable : bool = false
@export var hurtsOnImmune : bool = false
@export var isCollideable : bool = false
@export var Health : int = 1
@export var Ranged : bool = false
@export var GuardDistance : float = 40.0
@export var CircleRadius : float = 150.0
@export var RadiusTolerance : float = 30.0 
@export var hitbox : Area2D
var circleDir : float = 1.0

var timeSinceSpawn : float = 0.0
var hasAttacked : bool = false
var sprite2d : Sprite2D

const ARROW_SPEED = 260.0
const ARROW_LIFETIME = 2.0
const ARROW_MASK = 3 
func takeDamage(amount : int = 1) -> void:
	Health -= amount
	if Health <= 0:
		queue_free()

func _doDamage(body: Node2D):
	if body == self:
		return
		
	print("Hit registered on: ", body.name)
	if body.has_method("takeDamage") and not body.is_in_group("enemies"):
		body.takeDamage(1, [])

func _isGuarding() -> bool:
	return isBodyguard and (hasAttacked or DeadlineToAttack <= 0.0)

func _getNearestCelebrity() -> Enemy:
	var best : Enemy = null
	var best_dist := INF
	for node in get_tree().get_nodes_in_group("celebrities"):
		var celeb := node as Enemy
		if celeb == null or celeb == self:
			continue
		var d := global_position.distance_squared_to(celeb.global_position)
		if d < best_dist:
			best_dist = d
			best = celeb
	return best

func _shootArrow(direction: Vector2) -> void:
	var shape = RectangleShape2D.new()
	shape.size = Vector2(14, 5)
	var arrow_sprite = Sprite2D.new()
	arrow_sprite.texture = load("res://Enemies/arrow.png")
	arrow_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	arrow_sprite.scale = Vector2(1.75, 1.75)
	arrow_sprite.rotation = direction.angle()   # assumes arrow.png points right
	Hitbox._createProjectile(shape, ARROW_LIFETIME, direction * ARROW_SPEED, ARROW_MASK, arrow_sprite, global_position)

func _meleeAttack() -> void:
	var shape = RectangleShape2D.new()
	shape.size = Vector2(16, 16)
	var hitbox_area = Hitbox._createHitboxSqr(shape, 0.5, null, velocity, 2, null, global_position) as Area2D
	if hitbox_area:
		hitbox_area.body_entered.connect(_doDamage)
	var list = hitbox_area.get_overlapping_bodies()
	for i in range(list.size()):
		if not list[i].is_in_group("enemies"):
			if list[i].has_method("takeDamage"):
				list[i].takeDamage(1, ["Charging"])

func _ready() -> void:
	add_to_group("enemies")

	circleDir = 1.0 if randf() < 0.5 else -1.0
	if not Collision_Shape:
		Collision_Shape = CollisionShape2D.new()
		var body_shape = RectangleShape2D.new()
		body_shape.size = Vector2(18, 18)
		Collision_Shape.shape = body_shape
		
	if Collision_Shape.get_parent() == null:
		self.add_child(Collision_Shape)

	if not hitbox:
		hitbox = Area2D.new()
		self.add_child(hitbox)
		hitbox.collision_mask = 3
		var hitbox_cs = CollisionShape2D.new()
		var hitbox_shape = RectangleShape2D.new()
		hitbox_shape.size = Vector2(22, 22)
		hitbox_cs.shape = hitbox_shape
		hitbox.add_child(hitbox_cs)

	if isCollideable:
		collision_layer = 3
		collision_mask = 1
	else:
		collision_layer = 0
		collision_mask = 1

	var sprite = Sprite2D.new()
	self.add_child(sprite)
	if Sprite:
		sprite.texture = Sprite
	sprite2d = sprite
func _process(delta: float) -> void:
	timeSinceSpawn += delta
	
	var ints = Hitbox.find_ancestor_by_name(self, "main")
	if ints:
		var plr = ints.find_child("Slimey", true, false) as CharacterBody2D
		if plr:
			var to_player = plr.global_position - global_position
			if to_player.x > 0:
				sprite2d.flip_h = false
			else:
				sprite2d.flip_h = true
			var dist = to_player.length()
			var toward = to_player.normalized()
			var dir = Vector2.ZERO
			
			var celeb : Enemy = _getNearestCelebrity() if _isGuarding() else null

			if celeb:
				var celeb_to_plr = (plr.global_position - celeb.global_position).normalized()
				var guard_point = celeb.global_position + celeb_to_plr * GuardDistance
				var to_point = guard_point - global_position
				var arrive = clampf(to_point.length() / 40.0, 0.0, 1.0)
				dir = to_point.normalized() * arrive
			elif not Ranged:
				dir = toward
			else:
				var tangent = toward.orthogonal() * circleDir
				var radial = clampf((dist - CircleRadius) / RadiusTolerance, -1.0, 1.0)
				dir = (toward * radial + tangent * (1.0 - abs(radial) * 0.5)).normalized()

			velocity = dir * Speed
			move_and_slide()
			
			if Ranged and not celeb and is_on_wall():
				circleDir *= -1.0
			if DeadlineToAttack > 0.0 and timeSinceSpawn >= DeadlineToAttack and not hasAttacked:
				hasAttacked = true
				if Ranged:
					_shootArrow(toward)
					DeadlineToAttack = INF
				else:
					_meleeAttack()
		else:
			print("No Player found, err in enemy class")
			self.queue_free()
	else:
		print("Cannot find tree root/ancestor")

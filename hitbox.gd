extends Node

func find_ancestor_by_name(current_node: Node, target_name: String) -> Node:
	var p = current_node.get_parent()
	while p != null:
		if p.name == target_name:
			return p
		p = p.get_parent()
	return null

func find_ancestor_by_type(current_node: Node, type_class) -> Node:
	var p = current_node.get_parent()
	while p != null:
		if is_instance_of(p, type_class):
			return p
		p = p.get_parent()
	return null
	
func _initializeHitbox(shape_resource: Shape2D, parent: Node, layer : int, start_pos : Vector2 = Vector2.ZERO) -> Array:
	var crate = RigidBody2D.new()
	crate.gravity_scale = 0.0
	crate.linear_damp_mode = RigidBody2D.DAMP_MODE_REPLACE
	crate.linear_damp = 0.0
	crate.collision_layer = 0
	crate.collision_mask = 0

	var ins = Area2D.new()
	ins.collision_layer = 0
	ins.collision_mask = layer
	
	if parent:
		parent.add_child(crate)
	else:
		var tree := Engine.get_main_loop() as SceneTree
		tree.current_scene.add_child(crate)
	crate.global_position = start_pos

	crate.add_child(ins)
	var box = CollisionShape2D.new()
	ins.add_child(box)
	box.shape = shape_resource
	return [ins, box, crate]

func delayDeath(obj: Object, duration: float) -> void:
	var tree := Engine.get_main_loop() as SceneTree
	await tree.create_timer(duration).timeout
	if is_instance_valid(obj):
		obj.queue_free()

func _createHitboxCir(shape_resource: CircleShape2D, duration: float, parent: Node, velocity: Vector2, layer : int, sprite = null, start_pos : Vector2 = Vector2.ZERO):
	var tab = _initializeHitbox(shape_resource, parent, layer, start_pos)
	tab[2].linear_velocity = velocity
	if sprite:
		tab[2].add_child(sprite)
	delayDeath(tab[2], duration)
	return tab[0]

func _createHitboxSqr(shape_resource: RectangleShape2D, duration: float, parent: Node, velocity: Vector2, layer : int, sprite = null, start_pos : Vector2 = Vector2.ZERO):
	var tab = _initializeHitbox(shape_resource, parent, layer, start_pos)
	tab[2].linear_velocity = velocity
	if sprite:
		tab[2].add_child(sprite)
	delayDeath(tab[2], duration)
	return tab[0]

func _createProjectile(shape_resource: Shape2D, duration: float, velocity: Vector2, layer : int, sprite, start_pos : Vector2, damage : int = 1) -> Area2D:
	var area : Area2D
	if shape_resource is CircleShape2D:
		area = _createHitboxCir(shape_resource, duration, null, velocity, layer, sprite, start_pos)
	else:
		area = _createHitboxSqr(shape_resource, duration, null, velocity, layer, sprite, start_pos)
	area.rotation = velocity.angle()
	area.body_entered.connect(func(body: Node2D):
		if body is Enemy:
			return
		if body.has_method("takeDamage"):
			body.takeDamage(damage, [])
		else:
			print("Projectile hit: ", body.name)
		if is_instance_valid(area):
			area.get_parent().queue_free()
	)
	return area

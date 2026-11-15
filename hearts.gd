extends HBoxContainer

@export var player : CharacterBody2D
@export var full_heart : Texture2D
@export var broken_heart : Texture2D

func _ready() -> void:
	player.health_changed.connect(update_hearts)
	update_hearts(3,3)

func update_hearts(current: int, maximum: int) -> void:
	for child in get_children():
		child.queue_free()
	for i in maximum:
		var heart := TextureRect.new()
		heart.texture = full_heart if i < current else broken_heart
		heart.stretch_mode = TextureRect.STRETCH_KEEP
		add_child(heart)

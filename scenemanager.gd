extends Node

var loadpath = ""

# Called when the node enters the scene tree for the first time.
func _change_scene_amd_call(scene_path: String, loadpath : String):
	get_tree().call_deferred(scene_path, loadpath)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func setLoadpath(string : String):
	loadpath = string
	pass
	
func getLoadpath():
	return loadpath

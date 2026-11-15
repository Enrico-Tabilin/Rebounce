extends Node

var loadpath = ""

func change_scene(scene_path: String) -> void:
	get_tree().change_scene_to_file(scene_path)

func setLoadpath(string: String) -> void:
	loadpath = string
	
func getLoadpath() -> String:
	return loadpath
